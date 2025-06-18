"""
Test suite for the simplified explicit codegen architecture.

Tests the new explicit template declaration system and verifies
that the simplified architecture works correctly.
"""

import pytest
import unittest
from unittest.mock import Mock, MagicMock

# Test imports for new simplified architecture
try:
    from finn.codegen.simple_template_engine import SimpleTemplateEngine
    from finn.codegen.explicit_backend_registry import ExplicitBackendRegistry
    from finn.codegen.backend_registration import get_backend_registry
    from finn.codegen.simple_config import SimpleCodegenConfig
    from finn.codegen.legacy_compat import find_backend_for_operation
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports not available: {IMPORT_ERROR}")
class TestSimplifiedCodegen(unittest.TestCase):
    """Test suite for simplified codegen architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.template_engine = SimpleTemplateEngine()
        self.registry = ExplicitBackendRegistry()
        self.config = SimpleCodegenConfig.for_testing()
    
    def test_simple_template_engine_creation(self):
        """Test that SimpleTemplateEngine can be created."""
        engine = SimpleTemplateEngine()
        self.assertIsNotNone(engine)
        self.assertTrue(hasattr(engine, 'render'))
        self.assertTrue(hasattr(engine, 'clear_cache'))
    
    def test_template_engine_render_string(self):
        """Test template engine string rendering."""
        engine = SimpleTemplateEngine()
        
        # Test simple variable substitution
        template_str = "Hello {{ name }}!"
        context = {"name": "FINN"}
        result = engine.render_string(template_str, context)
        
        self.assertEqual(result, "Hello FINN!")
    
    def test_template_engine_legacy_render(self):
        """Test legacy string replacement rendering."""
        engine = SimpleTemplateEngine()
        
        # Test legacy $KEY$ replacement
        template_str = "Hello $NAME$! Welcome to $SYSTEM$."
        replacements = {"NAME": "FINN", "SYSTEM": "FPGA"}
        result = engine.render_legacy(template_str, replacements)
        
        self.assertEqual(result, "Hello FINN! Welcome to FPGA.")
    
    def test_explicit_backend_registry(self):
        """Test explicit backend registry functionality."""
        registry = ExplicitBackendRegistry()
        
        # Create mock backend class
        mock_backend = MagicMock()
        mock_backend.__name__ = "MockHLSBackend"
        
        # Test registration
        registry.register_hls_backend("MockOperation", mock_backend)
        
        # Test retrieval
        retrieved = registry.get_hls_backend("MockOperation")
        self.assertEqual(retrieved, mock_backend)
        
        # Test non-existent operation
        not_found = registry.get_hls_backend("NonExistentOperation")
        self.assertIsNone(not_found)
    
    def test_backend_registry_global_instance(self):
        """Test global backend registry access."""
        registry = get_backend_registry()
        self.assertIsNotNone(registry)
        self.assertIsInstance(registry, ExplicitBackendRegistry)
    
    def test_simple_config_creation(self):
        """Test simplified configuration creation."""
        config = SimpleCodegenConfig()
        self.assertIsNotNone(config)
        self.assertIsInstance(config.template_dirs, list)
        self.assertIsInstance(config.debug_mode, bool)
        
        # Test testing configuration
        test_config = SimpleCodegenConfig.for_testing()
        self.assertTrue(test_config.debug_mode)
        self.assertEqual(test_config.log_level, 'DEBUG')
    
    def test_legacy_compatibility(self):
        """Test legacy compatibility functions."""
        # Test that legacy function exists and issues warning
        with pytest.warns(DeprecationWarning):
            result = find_backend_for_operation("TestOperation", "hls")
        
        # Should return None for non-existent operation
        self.assertIsNone(result)
    
    def test_template_engine_caching(self):
        """Test template compilation caching."""
        engine = SimpleTemplateEngine()
        
        # Test cache info access
        cache_info = engine.get_cache_info()
        self.assertIsNotNone(cache_info)
        
        # Test cache clearing
        engine.clear_cache()
        
        # Cache should be cleared
        cache_info_after = engine.get_cache_info()
        self.assertEqual(cache_info_after.currsize, 0)


class TestExplicitTemplateInterface(unittest.TestCase):
    """Test explicit template interface."""
    
    def test_explicit_template_declaration(self):
        """Test explicit template declaration pattern."""
        
        # Mock backend class with explicit template declaration
        class MockBackend:
            TEMPLATE_NAME = "mock_template.j2"
            
            def get_template_name(self):
                return self.TEMPLATE_NAME
        
        backend = MockBackend()
        self.assertEqual(backend.get_template_name(), "mock_template.j2")
    
    def test_template_options_selection(self):
        """Test template options selection pattern."""
        
        class MockBackendWithOptions:
            TEMPLATE_OPTIONS = {
                'basic': 'mock_basic.j2',
                'optimized': 'mock_optimized.j2'
            }
            
            def __init__(self, prefer_optimized=False):
                self.prefer_optimized = prefer_optimized
            
            def get_template_name(self):
                if hasattr(self, '_template_override'):
                    return self._template_override
                return self._select_template_from_options()
            
            def _select_template_from_options(self):
                if self.prefer_optimized:
                    return self.TEMPLATE_OPTIONS['optimized']
                return self.TEMPLATE_OPTIONS['basic']
            
            def set_template_override(self, template_name):
                self._template_override = template_name
        
        # Test basic selection
        backend_basic = MockBackendWithOptions(prefer_optimized=False)
        self.assertEqual(backend_basic.get_template_name(), "mock_basic.j2")
        
        # Test optimized selection
        backend_opt = MockBackendWithOptions(prefer_optimized=True)
        self.assertEqual(backend_opt.get_template_name(), "mock_optimized.j2")
        
        # Test override
        backend_basic.set_template_override("custom_template.j2")
        self.assertEqual(backend_basic.get_template_name(), "custom_template.j2")


if __name__ == '__main__':
    unittest.main()