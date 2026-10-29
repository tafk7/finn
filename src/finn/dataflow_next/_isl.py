# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Private, thread-confined isl service. Durable/public values remain descriptors.

Only documented integer Map/Set algebra is used. Integer labels distinguish data
source slots (>=0) from fills (-1, with the fill value in the offset coordinate).
No printed relation or raw native hash is a semantic/durable key. Handles keep
their owning service/context alive and must be rebuilt for another thread.

Native substitution and equality do not extract coordinates. Export first uses
an independently verified descriptor certificate; bounded legacy canonical
extraction is explicit and never counted as enumeration removal.
"""

from __future__ import annotations

import islpy as isl
import numpy as np
from dataclasses import dataclass
from math import prod
from threading import Thread, current_thread
from typing import Literal

from finn.dataflow_next.maps import (
    MAX_PIECES,
    Piece,
    PositionMap,
    Unmappable,
    _partition,
    _strided_piece,
    _through_loops,
)
from finn.dataflow_next.traversal import (
    Grouped,
    Loop,
    Reorder,
    Traversal,
    axis_strides,
    grouped,
    one_pass,
)


def _digit(index: str, inner: int, extent: int) -> str:
    return f"(floor(({index})/{inner}) mod {extent})"


def _offset(index: str, loops: tuple[Loop, ...]) -> str:
    return (
        " + ".join(
            f"{_digit(index, inner, loop.extent)}*{loop.stride}"
            for loop, inner in zip(loops, axis_strides(tuple(loop.extent for loop in loops)))
        )
        or "0"
    )


@dataclass(frozen=True, eq=False)
class _Map:
    _service: _Service
    shape: tuple[int, ...]
    sources: tuple[tuple[int, ...], ...]
    source_ids: tuple[str, ...]
    _relation: isl.Map
    _certificate: PositionMap | None = None

    def __hash__(self) -> int:
        self._service._check(self)
        # Intentionally coarse. Equal wrappers hash alike; collisions use native equality.
        return hash((self.shape, self.sources, self.source_ids))

    def __eq__(self, other: object) -> bool:
        self._service._check(self)
        if not isinstance(other, _Map):
            return False
        self._service._check(other)
        return (self.shape, self.sources, self.source_ids) == (
            other.shape,
            other.sources,
            other.source_ids,
        ) and self._relation.is_equal(other._relation)


@dataclass(frozen=True)
class _Export:
    descriptor: PositionMap
    method: Literal["certified-descriptor", "bounded-legacy-extraction"]
    enumerated_positions: int = 0


@dataclass(frozen=True)
class _WordRoute:
    _service: _Service
    source_id: str
    frame_id: str
    pass_id: str
    set_id: str | None
    source: Traversal | Grouped
    sink: Traversal | Grouped
    _relation: isl.Map


class _Unsupported(ValueError):
    """Valid query outside supported export/resource policy; not an input refusal."""


class _Service:
    def __init__(self) -> None:
        self._context = isl.Context()
        self._thread: Thread = current_thread()

    def _check(self, *values: _Map | _WordRoute) -> None:
        if current_thread() is not self._thread:
            raise RuntimeError("rebuild descriptors in a service owned by this thread")
        if any(value._service is not self for value in values):
            raise RuntimeError("rebuild descriptors in this service's context")

    def _read(self, text: str) -> isl.Map:
        self._check()
        return isl.Map.read_from_str(self._context, text)

    def import_descriptor(
        self, descriptor: PositionMap, source_ids: tuple[str, ...] | None = None
    ) -> _Map:
        self._check()
        if not isinstance(descriptor, PositionMap):
            raise TypeError("import a validated PositionMap descriptor")
        ids = (
            tuple(str(i) for i in range(len(descriptor.sources)))
            if source_ids is None
            else tuple(source_ids)
        )
        if len(ids) != len(descriptor.sources) or not all(
            isinstance(name, str) and name for name in ids
        ):
            raise Unmappable("source-unknown", "import_descriptor", "name each ordered source slot")
        relation = self._read("{ [r] -> [s,o] : false }")
        occupied = isl.Set.read_from_str(self._context, "{ [r] : false }")
        for piece in descriptor.pieces:
            digits = [
                _digit("r", stride, extent)
                for stride, extent in zip(axis_strides(descriptor.shape), descriptor.shape)
            ]
            box = (
                " and ".join(
                    f"{start} <= {digit} < {start+extent}"
                    for digit, start, extent in zip(digits, piece.start, piece.extent)
                )
                or "true"
            )
            local = (
                " + ".join(
                    f"({digit}-{start})*{stride}"
                    for digit, start, stride in zip(digits, piece.start, axis_strides(piece.extent))
                )
                or "0"
            )
            body = self._read(
                f"{{ [r] -> [s,o] : 0 <= r < {prod(descriptor.shape)} and {box} "
                f"and s={piece.source} and o={piece.base}+({_offset(local, piece.modes)}) }}"
            )
            # Syntactic boxes reject overlaps even when their values agree.
            if not occupied.intersect(body.domain()).is_empty():
                raise Unmappable(
                    "pieces-overlap", "PositionMap", f"the box at {piece.start} overlaps"
                )
            occupied = occupied.union(body.domain())
            relation = relation.union(body)
        # Exact native simplification before complement/equality proofs. The
        # syntactic overlap checks above still reject identical input boxes;
        # native basic-map count never replaces canonical descriptor limits.
        occupied = occupied.coalesce()
        reader = isl.Set.read_from_str(
            self._context, f"{{ [r] : 0 <= r < {prod(descriptor.shape)} }}"
        )
        holes = reader.subtract(occupied)
        if not holes.is_empty():
            if descriptor.fill is None:
                raise Unmappable("fill-missing", "PositionMap", "uncovered reader positions")
            relation = relation.union(
                self._read(f"{{ [r] -> [s,o] : s=-1 and o={descriptor.fill} }}").intersect_domain(
                    holes
                )
            )
        relation = relation.coalesce()
        if not relation.is_single_valued() or not relation.domain().is_equal(reader):
            raise Unmappable(
                "map-invalid", "import_descriptor", "a map must be total and single-valued"
            )
        return _Map(self, descriptor.shape, descriptor.sources, ids, relation, descriptor)

    def equal(self, left: _Map, right: _Map) -> bool:
        self._check(left, right)
        return left == right

    def substitute(self, outer: _Map, inner: _Map, source: int = 0) -> _Map:
        """Read one selected inner source through outer; retain other slots and fills."""
        self._check(outer, inner)
        if type(source) is not int or not 0 <= source < len(inner.sources):
            raise Unmappable("source-unknown", "chain", f"source {source} of {len(inner.sources)}")
        if inner.sources[source] != outer.shape:
            raise Unmappable(
                "shape-mismatch", "chain", "the selected source must be the outer reader shape"
            )
        sources = (*inner.sources[:source], *outer.sources, *inner.sources[source + 1 :])
        ids = (*inner.source_ids[:source], *outer.source_ids, *inner.source_ids[source + 1 :])
        added = len(outer.sources) - 1
        select = self._read(f"{{ [s,o] -> [r] : s={source} and r=o }}")
        inserted = select.apply_range(outer._relation).apply_range(
            self._read(
                f"{{ [s,o] -> [t,p] : p=o and ((s>=0 and t=s+{source}) or (s=-1 and t=-1)) }}"
            )
        )
        retained = self._read(
            f"{{ [s,o] -> [t,p] : p=o and ((s<{source} and t=s) or (s>{source} and t=s+{added})) }}"
        )
        relation = inner._relation.apply_range(inserted.union(retained))
        fills = relation.range().intersect(isl.Set.read_from_str(self._context, "{ [s,o] : s=-1 }"))
        if not fills.is_empty() and not fills.dim_min_val(1).eq(fills.dim_max_val(1)):
            raise Unmappable("fills-differ", "chain", "positions read different fills")
        result = _Map(self, inner.shape, sources, ids, relation)
        certificate = self._composition_certificate(outer, inner, result, source)
        return _Map(self, result.shape, result.sources, result.source_ids, relation, certificate)

    def _composition_certificate(
        self, outer: _Map, inner: _Map, result: _Map, source: int
    ) -> PositionMap | None:
        a, b = outer._certificate, inner._certificate
        if a is None or b is None:
            return None
        # Identity substitutions retain compact disjoint pad/drop/concat certificates.
        if (
            len(b.sources) == 1
            and b.strided
            and b.pieces[0].modes == (() if prod(b.shape) == 1 else (Loop(prod(b.shape), 1),))
            and b.shape == a.shape
        ):
            return a
        if (
            len(a.sources) == 1
            and a.strided
            and a.shape == a.sources[0]
            and a.pieces[0].modes == (() if prod(a.shape) == 1 else (Loop(prod(a.shape), 1),))
        ):
            return b
        if not a.strided or not b.strided:
            return None
        try:
            modes = _through_loops(b.pieces[0].modes, a.pieces[0].modes, "chain")
            candidate = PositionMap(
                b.shape,
                result.sources,
                (
                    Piece(
                        (0,) * len(b.shape), b.shape, source + a.pieces[0].source, 0, tuple(modes)
                    ),
                ),
            )
            if self.import_descriptor(candidate, result.source_ids)._relation.is_equal(
                result._relation
            ):
                return candidate
        except (Unmappable, ValueError):
            return None
        return None

    def export_descriptor(
        self, handle: _Map, candidate: PositionMap | None = None, *, budget: int = 0
    ) -> _Export:
        self._check(handle)
        if type(budget) is not int or budget < 0 or budget > 65536:
            raise ValueError("legacy extraction budget is an integer from 0 to 65536 positions")
        candidate = candidate if candidate is not None else handle._certificate
        if candidate is not None:
            proposed = self.import_descriptor(candidate, handle.source_ids)
            if not self.equal(handle, proposed):
                raise Unmappable(
                    "export-differs",
                    "export_descriptor",
                    "candidate changes source positions or fills",
                )
            if len(candidate.pieces) > MAX_PIECES:
                raise Unmappable(
                    "too-many-pieces", "export_descriptor", "canonical piece bound exceeded"
                )
            return _Export(candidate, "certified-descriptor")
        size = prod(handle.shape)
        if size > budget:
            raise _Unsupported("no certified descriptor; bounded legacy extraction unavailable")
        # The retained canonicalizer uses signed int64 source labels/offsets.
        # Native/certified descriptors have no such limit. Fills stay Python
        # integers and enter these arrays only as the -1 sentinel.
        bound = (1 << 63) - 1
        unsafe = handle._relation.range().intersect(
            isl.Set.read_from_str(
                self._context,
                f"{{ [s,o] : s>=0 and (s>{bound} or o>{bound} or o<{-bound-1}) }}",
            )
        )
        if not unsafe.is_empty():
            raise _Unsupported(
                "bounded legacy extraction requires data source slots and offsets "
                "to fit signed int64"
            )
        # Explicit retained extraction/canonicalization, not a native path claim.
        label, value = np.empty(size, dtype=np.int64), np.empty(size, dtype=np.int64)
        fill: int | None = None
        for r in range(size):
            at = (
                handle._relation.fix_val(isl.dim_type.in_, 0, isl.Val.int_from_si(self._context, r))
                .range()
                .sample_point()
            )
            slot = int(at.get_coordinate_val(isl.dim_type.set, 0).to_python())
            offset = int(at.get_coordinate_val(isl.dim_type.set, 1).to_python())
            label[r], value[r] = slot, offset if slot >= 0 else -1
            if slot < 0:
                fill = offset
        pieces = _partition(label.reshape(handle.shape), value.reshape(handle.shape), "chain")
        candidate = PositionMap(handle.shape, handle.sources, tuple(pieces), fill)
        if not self.equal(handle, self.import_descriptor(candidate, handle.source_ids)):
            raise Unmappable(
                "export-differs", "export_descriptor", "legacy extraction changed the relation"
            )
        return _Export(candidate, "bounded-legacy-extraction", size)

    def _form(self, form: Traversal | Grouped, *, qualify_pass: bool = False) -> isl.Map:
        if not isinstance(form, (Traversal, Grouped)):
            raise TypeError("a form is a Traversal or Grouped")
        expr = (
            _offset(f"b*{form.lanes}+l", form.order)
            if isinstance(form, Grouped)
            else f"({_offset('b', form.beat_loops)})+({_offset('l', form.lane_loops)})"
        )
        target = "[p,o]" if qualify_pass else "[o]"
        qualification = f" and p=floor(b/{one_pass(form).beats})" if qualify_pass else ""
        return self._read(
            f"{{ [b,l] -> {target} : 0<=b<{form.beats} and 0<=l<{form.lanes} "
            f"and o={expr}{qualification} }}"
        )

    def verify_through(
        self, handle: _Map, form: Traversal | Grouped, candidate: Traversal | Grouped
    ) -> bool:
        self._check(handle)
        piece = _strided_piece(self.export_descriptor(handle).descriptor, "through")
        if form.shape != handle.shape or candidate.shape != handle.sources[piece.source]:
            return False
        composed = self._form(form).apply_range(handle._relation)
        proposed = self._form(candidate).apply_range(
            self._read(f"{{ [o] -> [s,p] : s={piece.source} and p=o }}")
        )
        return composed.is_equal(proposed)

    def presented_fills(
        self,
        descriptor: PositionMap,
        form: Traversal | Grouped,
        stage: PositionMap | None = None,
        source: int = 0,
    ) -> tuple[int, ...]:
        """Reachable reader values for typed-edge validation, without extracting a map.

        Inspect every lane of the reader and only the selected source's stage.
        This does not normalize/substitute a new public map or bypass its fill,
        factoring or export refusals. Unread fill definitions impose no value
        restriction on a reader's encoding.
        """
        self._check()
        if form.shape != descriptor.shape:
            raise Unmappable("shape-mismatch", "presented_fills", "reader shape differs")
        presented = self._form(form).apply_range(self.import_descriptor(descriptor)._relation)
        fill_range = isl.Set.read_from_str(self._context, "{ [s,o] : s=-1 }")
        found: set[int] = set()
        if descriptor.fill is not None and not presented.range().intersect(fill_range).is_empty():
            found.add(descriptor.fill)
        if stage is not None:
            if type(source) is not int or not 0 <= source < len(descriptor.sources):
                raise Unmappable("source-unknown", "presented_fills", "unknown stage source")
            if stage.shape != descriptor.sources[source] or len(stage.sources) != 1:
                raise Unmappable("shape-mismatch", "presented_fills", "stage does not match reader")
            selected = presented.apply_range(self._read(f"{{ [s,o] -> [r] : s={source} and r=o }}"))
            staged = selected.apply_range(self.import_descriptor(stage)._relation)
            if stage.fill is not None and not staged.range().intersect(fill_range).is_empty():
                found.add(stage.fill)
        return tuple(sorted(found))

    def through(self, handle: _Map, form: Traversal | Grouped) -> Traversal | Grouped:
        self._check(handle)
        if form.shape != handle.shape:
            raise Unmappable("shape-mismatch", "through", "the traversal must walk the map reader")
        descriptor = self.export_descriptor(handle).descriptor
        piece = _strided_piece(descriptor, "through")
        source = handle.sources[piece.source]
        if not source:
            raise Unmappable(
                "source-rank-zero", "through", "a traversal walks a tensor of rank one or more"
            )
        loops = form.order if isinstance(form, Grouped) else form.beat_loops
        lanes = () if isinstance(form, Grouped) else form.lane_loops
        beats = _through_loops(loops, piece.modes, "through")
        lane_loops = _through_loops(lanes, piece.modes, "through")
        try:
            result: Traversal | Grouped = Traversal(source, beats, lane_loops)
            if isinstance(form, Grouped):
                result = grouped(result, form.lanes)
        except ValueError as error:
            raise Unmappable("carry-between-loops", "through", str(error)) from error
        if not self.verify_through(handle, form, result):
            raise Unmappable(
                "carry-between-loops",
                "through",
                "the loops' strides carry into one another: the composition is no loop nest",
            )
        return result

    def beat_route(
        self,
        source: Traversal | Grouped,
        sink: Traversal | Grouped,
        *,
        source_id: str,
        frame_id: str,
        pass_id: str,
        set_id: str | None = None,
    ) -> _WordRoute:
        """First whole source word within the explicitly named frame/pass/set scope.

        A repeated element can reuse its first word inside this scope, preserving
        current slots policy. Cross-pass/set association is never inferred here.
        All lanes must agree, not merely lane zero. Markers stay on descriptors.
        """
        self._check()
        if not all(isinstance(v, str) and v for v in (source_id, frame_id, pass_id)) or (
            set_id is not None and (not isinstance(set_id, str) or not set_id)
        ):
            raise ValueError("word routes require explicit source/frame/pass/set identity")
        if source.shape != sink.shape:
            raise Unmappable("shape-mismatch", "beat_route", "word routes retain tensor shape")
        if source.beats // one_pass(source).beats < sink.beats // one_pass(sink).beats:
            raise _Unsupported("finite source descriptor does not bound every requested whole pass")
        if source.lanes != sink.lanes or (
            isinstance(source, Traversal)
            and isinstance(sink, Traversal)
            and source.lane_loops != sink.lane_loops
        ):
            raise Unmappable(
                "lanes-differ", "beat_route", "a reorder moves whole words with agreeing lanes"
            )
        produced = self._form(source, qualify_pass=True)
        wanted = self._form(sink, qualify_pass=True)
        matches = wanted.apply_range(produced.reverse()).intersect(
            self._read("{ [b,l] -> [q,k] : l=k }")
        )
        zero = isl.Val.zero(self._context)
        first = (
            matches.fix_val(isl.dim_type.in_, 1, zero)
            .project_out(isl.dim_type.in_, 1, 1)
            .project_out(isl.dim_type.out, 1, 1)
            .lexmin()
        )
        # Lift the first lane's beat association and prove every lane is delivered.
        chosen = (
            self._read("{ [b,l] -> [b] }")
            .apply_range(first)
            .flat_range_product(self._read("{ [b,l] -> [l] }"))
            .intersect_domain(wanted.domain())
        )
        if not chosen.apply_range(produced).is_equal(wanted):
            raise Unmappable(
                "element-never-presented",
                "beat_route",
                "no first whole source word delivers every requested lane",
            )
        return _WordRoute(self, source_id, frame_id, pass_id, set_id, source, sink, first)

    def reorder_route(self, reorder: Reorder) -> isl.Map:
        self._check()
        if (
            type(reorder.frame_beats) is not int
            or reorder.frame_beats < 1
            or (
                len(reorder.dims) != len(reorder.coefs)
                or any(type(v) is not int or v < 1 for v in reorder.dims)
                or any(type(v) is not int or v < 0 for v in reorder.coefs)
            )
        ):
            raise ValueError("a reorder has positive frame/extents and nonnegative strides")
        if sum((d - 1) * c for d, c in zip(reorder.dims, reorder.coefs)) >= reorder.frame_beats:
            raise ValueError("a reorder leaves its source frame")
        loops = tuple(Loop(d, c) for d, c in zip(reorder.dims, reorder.coefs))
        return self._read(
            f"{{ [i] -> [q] : 0<=i<{prod(reorder.dims)} and q={_offset('i',loops)} }}"
        )

    def order_bound(self, route: _WordRoute) -> int:
        self._check(route)
        return self._order_bound(route._relation)

    def word_at(self, route: _WordRoute, beat: int) -> int:
        self._check(route)
        if type(beat) is not int or not 0 <= beat < route.sink.beats:
            raise ValueError("a beat index must be within the finite sink sequence")
        point = (
            route._relation.fix_val(
                isl.dim_type.in_, 0, isl.Val.read_from_str(self._context, str(beat))
            )
            .range()
            .sample_point()
        )
        return int(point.get_coordinate_val(isl.dim_type.set, 0).to_python())

    def reorder_bound(self, reorder: Reorder) -> int:
        return self._order_bound(self.reorder_route(reorder))

    def _order_bound(self, relation: isl.Map) -> int:
        self._check()
        if relation.is_empty():
            return 0
        pairs = (
            relation.product(relation).flatten().intersect(self._read("{ [i,j] -> [a,b] : i<j }"))
        )
        if pairs.is_empty():
            return 1
        gaps = pairs.apply_range(self._read("{ [a,b] -> [gap] : gap=a-b }")).range()
        largest = gaps.dim_max_val(0)
        if not largest.is_int():
            raise _Unsupported("order bound is not a finite scalar integer")
        return 1 + max(0, int(largest.to_python()))
