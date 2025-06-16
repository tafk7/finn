def is_hls_node(node):
    """Returns True if given node is hls node. Otherwise False."""
    is_node = False
    if node is not None:
        # Original FINN HLS node pattern
        if node.domain.endswith(".custom_op.fpgadataflow.hls"):
            n_backend = get_by_name(node.attribute, "backend")
            if n_backend is not None:
                backend_value = n_backend.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    is_node = True
        # BrainSmith HLS node pattern - recognize brainsmith.libraries.kernels.* domains
        elif node.domain.startswith("brainsmith.libraries.kernels."):
            n_backend = get_by_name(node.attribute, "backend")
            if n_backend is not None:
                backend_value = n_backend.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    is_node = True

    return is_node


def is_rtl_node(node):
    """Returns True if given node is rtl node. Otherwise False."""
    is_node = False
    if node is not None:
        # Original FINN RTL node pattern
        if node.domain.endswith(".custom_op.fpgadataflow.rtl"):
            n_backend = get_by_name(node.attribute, "backend")
            if n_backend is not None:
                backend_value = n_backend.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    is_node = True
        # BrainSmith RTL node pattern - recognize brainsmith.libraries.kernels.* domains
        # (for future RTL implementations)
        elif node.domain.startswith("brainsmith.libraries.kernels.") and node.domain.endswith(".rtl"):
            n_backend = get_by_name(node.attribute, "backend")
            if n_backend is not None:
                backend_value = n_backend.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    is_node = True

    return is_node
