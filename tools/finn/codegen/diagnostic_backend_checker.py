#!/usr/bin/env python3

"""
🔍 Diagnostic Backend Checker
Analyzes backend implementations to identify missing methods and interface issues
"""

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# Setup logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackendDiagnostic:
    """Diagnoses backend implementation issues"""
    
    def __init__(self):
        logger.info("🔍 Initializing Backend Diagnostic Tool")
    
    def analyze_abstract_methods(self, backend_class) -> Dict[str, Any]:
        """Check for missing abstract methods"""
        logger.info(f"🔍 Analyzing abstract methods for {backend_class.__name__}")
        
        result = {
            'class_name': backend_class.__name__,
            'is_abstract': inspect.isabstract(backend_class),
            'abstract_methods': [],
            'all_methods': [],
            'mro': [cls.__name__ for cls in backend_class.__mro__]
        }
        
        # Get all abstract methods
        if hasattr(backend_class, '__abstractmethods__'):
            result['abstract_methods'] = list(backend_class.__abstractmethods__)
            logger.info(f"   📋 Abstract methods: {result['abstract_methods']}")
        
        # Get all methods
        result['all_methods'] = [name for name, method in inspect.getmembers(backend_class, inspect.ismethod)]
        
        # Check if class can be instantiated
        try:
            # Try with minimal args - we won't actually instantiate but see what happens
            sig = inspect.signature(backend_class.__init__)
            logger.info(f"   📋 Constructor signature: {sig}")
        except Exception as e:
            logger.warning(f"   ⚠️ Constructor analysis failed: {e}")
            
        return result
    
    def analyze_method_signatures(self, backend_class, method_names: List[str]) -> Dict[str, Any]:
        """Analyze specific method signatures"""
        logger.info(f"🔍 Analyzing method signatures for {backend_class.__name__}")
        
        result = {
            'class_name': backend_class.__name__,
            'methods': {}
        }
        
        for method_name in method_names:
            if hasattr(backend_class, method_name):
                method = getattr(backend_class, method_name)
                try:
                    sig = inspect.signature(method)
                    result['methods'][method_name] = {
                        'exists': True,
                        'signature': str(sig),
                        'parameters': list(sig.parameters.keys())
                    }
                    logger.info(f"   ✅ {method_name}: {sig}")
                except Exception as e:
                    result['methods'][method_name] = {
                        'exists': True,
                        'error': str(e)
                    }
                    logger.warning(f"   ⚠️ {method_name}: Error analyzing - {e}")
            else:
                result['methods'][method_name] = {'exists': False}
                logger.warning(f"   ❌ {method_name}: Method not found")
        
        return result
    
    def analyze_inheritance_chain(self, backend_class) -> Dict[str, Any]:
        """Analyze inheritance and base classes"""
        logger.info(f"🔍 Analyzing inheritance chain for {backend_class.__name__}")
        
        result = {
            'class_name': backend_class.__name__,
            'mro': [],
            'base_classes': [],
            'abstract_bases': []
        }
        
        # Method Resolution Order
        for cls in backend_class.__mro__:
            class_info = {
                'name': cls.__name__,
                'module': cls.__module__,
                'is_abstract': inspect.isabstract(cls),
                'abstract_methods': list(getattr(cls, '__abstractmethods__', []))
            }
            result['mro'].append(class_info)
            logger.info(f"   📋 {cls.__name__} ({'abstract' if class_info['is_abstract'] else 'concrete'})")
            if class_info['abstract_methods']:
                logger.info(f"       Abstract methods: {class_info['abstract_methods']}")
        
        return result
    
    def diagnose_node_object(self, node_obj) -> Dict[str, Any]:
        """Analyze the structure of a node object"""
        logger.info(f"🔍 Analyzing node object: {type(node_obj)}")
        
        result = {
            'type': str(type(node_obj)),
            'attributes': [],
            'methods': [],
            'has_name': False,
            'name_value': None
        }
        
        # Check if it's a string (common mistake)
        if isinstance(node_obj, str):
            logger.warning(f"   ⚠️ Node is a STRING: '{node_obj}' - This is likely the problem!")
            result['is_string'] = True
            return result
        
        # Get attributes and methods
        for attr_name in dir(node_obj):
            if not attr_name.startswith('_'):
                attr_value = getattr(node_obj, attr_name, None)
                if callable(attr_value):
                    result['methods'].append(attr_name)
                else:
                    result['attributes'].append(attr_name)
        
        # Check for name attribute specifically
        if hasattr(node_obj, 'name'):
            result['has_name'] = True
            result['name_value'] = getattr(node_obj, 'name', None)
            logger.info(f"   ✅ Has 'name' attribute: {result['name_value']}")
        else:
            logger.warning(f"   ❌ Missing 'name' attribute")
        
        logger.info(f"   📋 Attributes: {result['attributes'][:10]}...")  # Show first 10
        logger.info(f"   📋 Methods: {result['methods'][:10]}...")      # Show first 10
        
        return result

def main():
    """Run comprehensive backend diagnostics"""
    logger.info("🚀 Starting Comprehensive Backend Diagnostic")
    
    diagnostic = BackendDiagnostic()
    
    print("=" * 80)
    print("🔍 BACKEND DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    
    # Test importing backends
    try:
        print("\n📦 Testing Backend Imports...")
        
        # Import clean backends
        from finn.custom_op.fpgadataflow.hls.CG_thresholding_hls import CG_Thresholding_hls
        from finn.custom_op.fpgadataflow.hls.CG_mvau_hls import CG_MVAU_hls
        
        # Import legacy backends  
        from finn.custom_op.fpgadataflow.hls.thresholding_hls import Thresholding_hls
        from finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls import MVAU_hls
        
        print("✅ All backends imported successfully")
        
        backends_to_analyze = [
            ("Clean Thresholding", CG_Thresholding_hls),
            ("Clean MVAU", CG_MVAU_hls), 
            ("Legacy Thresholding", Thresholding_hls),
            ("Legacy MVAU", MVAU_hls)
        ]
        
        for name, backend_class in backends_to_analyze:
            print(f"\n🔍 ANALYZING: {name}")
            print("-" * 50)
            
            # Abstract method analysis
            abstract_result = diagnostic.analyze_abstract_methods(backend_class)
            print(f"Abstract: {abstract_result['is_abstract']}")
            print(f"Missing methods: {abstract_result['abstract_methods']}")
            
            # Method signature analysis
            key_methods = [
                'get_template_values',
                'code_generation_cppsim',
                '_generate_common_values',
                '_generate_operation_specific_values'
            ]
            method_result = diagnostic.analyze_method_signatures(backend_class, key_methods)
            
            for method, info in method_result['methods'].items():
                if info['exists']:
                    print(f"✅ {method}: {info.get('signature', 'OK')}")
                else:
                    print(f"❌ {method}: MISSING")
            
            # Inheritance analysis
            inherit_result = diagnostic.analyze_inheritance_chain(backend_class)
            print(f"Inheritance chain: {' -> '.join([cls['name'] for cls in inherit_result['mro'][:3]])}...")
        
        # Test node object creation
        print(f"\n🔍 TESTING NODE OBJECT CREATION")
        print("-" * 50)
        
        from finn.codegen.test_node_factory import TestNodeFactory
        factory = TestNodeFactory()
        
        # Test thresholding node
        thresh_node = factory.create_thresholding_node()
        thresh_analysis = diagnostic.diagnose_node_object(thresh_node)
        print(f"Thresholding node type: {thresh_analysis['type']}")
        print(f"Has name attribute: {thresh_analysis['has_name']}")
        if thresh_analysis['has_name']:
            print(f"Name value: {thresh_analysis['name_value']}")
        
        # Test MVAU node
        mvau_node = factory.create_mvau_node() 
        mvau_analysis = diagnostic.diagnose_node_object(mvau_node)
        print(f"MVAU node type: {mvau_analysis['type']}")
        print(f"Has name attribute: {mvau_analysis['has_name']}")
        if mvau_analysis['has_name']:
            print(f"Name value: {mvau_analysis['name_value']}")
            
    except Exception as e:
        logger.error(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏆 DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()