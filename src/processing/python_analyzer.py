"""
Python AST analyzer for semantic code analysis.

This module provides advanced Python code analysis using the Abstract Syntax Tree (AST)
to extract semantic information, dependencies, and code structure.
"""

import ast
import hashlib
import re
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logger import get_logger
from src.utils.text_utils import clean_text

logger = get_logger(__name__)


class CodeElementType(Enum):
    """Types of Python code elements."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    COMMENT = "comment"
    DOCSTRING = "docstring"


@dataclass
class CodeElement:
    """Represents a Python code element with metadata."""
    name: str
    element_type: CodeElementType
    start_line: int
    end_line: int
    content: str
    docstring: Optional[str] = None
    parent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def line_count(self) -> int:
        """Number of lines in this element."""
        return self.end_line - self.start_line + 1
    
    @property
    def complexity_score(self) -> int:
        """Basic complexity score based on content."""
        score = 0
        # Count control flow statements
        score += len(re.findall(r'\b(if|elif|else|for|while|try|except|finally|with)\b', self.content))
        # Count function/method calls
        score += len(re.findall(r'\w+\s*\(', self.content))
        # Count nested structures
        score += self.content.count('    ')  # Basic indentation counting
        return score


@dataclass
class DependencyInfo:
    """Information about code dependencies."""
    imports: List[str] = field(default_factory=list)
    from_imports: Dict[str, List[str]] = field(default_factory=dict)
    function_calls: List[str] = field(default_factory=list)
    class_references: List[str] = field(default_factory=list)
    variable_references: List[str] = field(default_factory=list)


@dataclass
class CodeAnalysis:
    """Complete analysis of Python code."""
    elements: List[CodeElement] = field(default_factory=list)
    dependencies: DependencyInfo = field(default_factory=DependencyInfo)
    metrics: Dict[str, Any] = field(default_factory=dict)
    syntax_errors: List[str] = field(default_factory=list)
    
    def get_elements_by_type(self, element_type: CodeElementType) -> List[CodeElement]:
        """Get all elements of a specific type."""
        return [elem for elem in self.elements if elem.element_type == element_type]
    
    def get_element_by_name(self, name: str) -> Optional[CodeElement]:
        """Get element by name."""
        for elem in self.elements:
            if elem.name == name:
                return elem
        return None


class PythonASTAnalyzer:
    """
    Advanced Python code analyzer using AST.
    
    Features:
    - Semantic code structure extraction
    - Dependency analysis and mapping
    - Code quality metrics calculation
    - Error detection and reporting
    - Documentation extraction
    """
    
    def __init__(self):
        """Initialize the AST analyzer."""
        self.logger = get_logger(__name__, analyzer="python_ast")
    
    def analyze_code(self, code: str, filename: str = "<string>") -> CodeAnalysis:
        """
        Analyze Python code and extract semantic information.
        
        Args:
            code: Python source code to analyze
            filename: Name of the source file (for error reporting)
            
        Returns:
            CodeAnalysis: Complete analysis results
        """
        analysis = CodeAnalysis()
        
        try:
            # Parse the code into AST
            tree = ast.parse(code, filename=filename)
            
            # Extract code elements
            analysis.elements = self._extract_elements(tree, code)
            
            # Analyze dependencies
            analysis.dependencies = self._analyze_dependencies(tree)
            
            # Calculate metrics
            analysis.metrics = self._calculate_metrics(tree, code, analysis.elements)
            
            self.logger.debug(
                "Code analysis completed",
                filename=filename,
                elements_found=len(analysis.elements),
                lines_of_code=len(code.splitlines())
            )
            
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            analysis.syntax_errors.append(error_msg)
            self.logger.warning(
                "Syntax error in code analysis",
                filename=filename,
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            analysis.syntax_errors.append(error_msg)
            self.logger.error(
                "Unexpected error in code analysis",
                filename=filename,
                error=error_msg
            )
        
        return analysis
    
    def _extract_elements(self, tree: ast.AST, code: str) -> List[CodeElement]:
        """Extract code elements from AST."""
        elements = []
        code_lines = code.splitlines()
        
        class ElementVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_class = None
                self.elements = []
            
            def visit_ClassDef(self, node):
                """Visit class definition."""
                element = self._create_element(
                    node, CodeElementType.CLASS, code_lines, self.current_class
                )
                self.elements.append(element)
                
                # Set current class for methods
                old_class = self.current_class
                self.current_class = node.name
                
                # Visit child nodes
                self.generic_visit(node)
                
                # Restore previous class
                self.current_class = old_class
            
            def visit_FunctionDef(self, node):
                """Visit function/method definition."""
                element_type = (
                    CodeElementType.METHOD if self.current_class 
                    else CodeElementType.FUNCTION
                )
                
                element = self._create_element(
                    node, element_type, code_lines, self.current_class
                )
                self.elements.append(element)
                
                # Visit child nodes
                self.generic_visit(node)
            
            def visit_AsyncFunctionDef(self, node):
                """Visit async function definition."""
                self.visit_FunctionDef(node)  # Same handling as regular function
            
            def _create_element(self, node, element_type, code_lines, parent):
                """Create a CodeElement from an AST node."""
                start_line = node.lineno
                end_line = getattr(node, 'end_lineno', start_line)
                
                # Extract content
                if end_line and start_line <= len(code_lines):
                    content_lines = code_lines[start_line-1:end_line]
                    content = '\n'.join(content_lines)
                else:
                    content = ""
                
                # Extract docstring
                docstring = ast.get_docstring(node)
                
                # Create metadata
                metadata = {
                    'args': [],
                    'decorators': [],
                    'returns': None,
                    'is_async': isinstance(node, ast.AsyncFunctionDef)
                }
                
                # Add function-specific metadata
                if hasattr(node, 'args'):
                    metadata['args'] = [arg.arg for arg in node.args.args]
                
                if hasattr(node, 'decorator_list'):
                    metadata['decorators'] = [
                        ast.unparse(dec) if hasattr(ast, 'unparse') else str(dec)
                        for dec in node.decorator_list
                    ]
                
                if hasattr(node, 'returns') and node.returns:
                    metadata['returns'] = (
                        ast.unparse(node.returns) if hasattr(ast, 'unparse')
                        else str(node.returns)
                    )
                
                return CodeElement(
                    name=node.name,
                    element_type=element_type,
                    start_line=start_line,
                    end_line=end_line or start_line,
                    content=content,
                    docstring=docstring,
                    parent=parent,
                    metadata=metadata
                )
        
        visitor = ElementVisitor()
        visitor.visit(tree)
        
        return visitor.elements
    
    def _analyze_dependencies(self, tree: ast.AST) -> DependencyInfo:
        """Analyze code dependencies."""
        dependencies = DependencyInfo()
        
        class DependencyVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                """Visit import statement."""
                for alias in node.names:
                    dependencies.imports.append(alias.name)
            
            def visit_ImportFrom(self, node):
                """Visit from-import statement."""
                module = node.module or ""
                names = [alias.name for alias in node.names]
                
                if module in dependencies.from_imports:
                    dependencies.from_imports[module].extend(names)
                else:
                    dependencies.from_imports[module] = names
            
            def visit_Call(self, node):
                """Visit function call."""
                if isinstance(node.func, ast.Name):
                    dependencies.function_calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Handle method calls like obj.method()
                    if hasattr(ast, 'unparse'):
                        dependencies.function_calls.append(ast.unparse(node.func))
                
                self.generic_visit(node)
            
            def visit_Name(self, node):
                """Visit name reference."""
                if isinstance(node.ctx, ast.Load):
                    dependencies.variable_references.append(node.id)
        
        visitor = DependencyVisitor()
        visitor.visit(tree)
        
        # Remove duplicates
        dependencies.imports = list(set(dependencies.imports))
        dependencies.function_calls = list(set(dependencies.function_calls))
        dependencies.variable_references = list(set(dependencies.variable_references))
        
        return dependencies
    
    def _calculate_metrics(self, tree: ast.AST, code: str, elements: List[CodeElement]) -> Dict[str, Any]:
        """Calculate code quality metrics."""
        lines = code.splitlines()
        
        metrics = {
            'total_lines': len(lines),
            'blank_lines': sum(1 for line in lines if not line.strip()),
            'comment_lines': sum(1 for line in lines if line.strip().startswith('#')),
            'code_lines': 0,
            'classes': len([e for e in elements if e.element_type == CodeElementType.CLASS]),
            'functions': len([e for e in elements if e.element_type == CodeElementType.FUNCTION]),
            'methods': len([e for e in elements if e.element_type == CodeElementType.METHOD]),
            'complexity_score': 0,
            'max_line_length': max(len(line) for line in lines) if lines else 0,
            'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0
        }
        
        # Calculate code lines (non-blank, non-comment)
        metrics['code_lines'] = (
            metrics['total_lines'] - 
            metrics['blank_lines'] - 
            metrics['comment_lines']
        )
        
        # Calculate complexity score
        metrics['complexity_score'] = sum(elem.complexity_score for elem in elements)
        
        # Calculate maintainability metrics
        if metrics['code_lines'] > 0:
            metrics['comment_ratio'] = metrics['comment_lines'] / metrics['code_lines']
            metrics['function_density'] = (
                (metrics['functions'] + metrics['methods']) / metrics['code_lines'] * 100
            )
        else:
            metrics['comment_ratio'] = 0
            metrics['function_density'] = 0
        
        return metrics
    
    def extract_imports(self, code: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Extract import statements from code.
        
        Args:
            code: Python source code
            
        Returns:
            Tuple of (imports, from_imports)
        """
        try:
            tree = ast.parse(code)
            dependencies = self._analyze_dependencies(tree)
            return dependencies.imports, dependencies.from_imports
        except Exception as e:
            self.logger.warning(f"Failed to extract imports: {e}")
            return [], {}
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract function/method information from code.
        
        Args:
            code: Python source code
            
        Returns:
            List of function information dictionaries
        """
        try:
            analysis = self.analyze_code(code)
            functions = []
            
            for element in analysis.elements:
                if element.element_type in [CodeElementType.FUNCTION, CodeElementType.METHOD]:
                    functions.append({
                        'name': element.name,
                        'type': element.element_type.value,
                        'start_line': element.start_line,
                        'end_line': element.end_line,
                        'line_count': element.line_count,
                        'docstring': element.docstring,
                        'parent_class': element.parent,
                        'args': element.metadata.get('args', []),
                        'decorators': element.metadata.get('decorators', []),
                        'is_async': element.metadata.get('is_async', False),
                        'complexity_score': element.complexity_score
                    })
            
            return functions
            
        except Exception as e:
            self.logger.warning(f"Failed to extract functions: {e}")
            return []
    
    def extract_classes(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract class information from code.
        
        Args:
            code: Python source code
            
        Returns:
            List of class information dictionaries
        """
        try:
            analysis = self.analyze_code(code)
            classes = []
            
            for element in analysis.elements:
                if element.element_type == CodeElementType.CLASS:
                    # Get methods in this class
                    methods = [
                        e for e in analysis.elements 
                        if e.element_type == CodeElementType.METHOD and e.parent == element.name
                    ]
                    
                    classes.append({
                        'name': element.name,
                        'start_line': element.start_line,
                        'end_line': element.end_line,
                        'line_count': element.line_count,
                        'docstring': element.docstring,
                        'method_count': len(methods),
                        'methods': [m.name for m in methods],
                        'decorators': element.metadata.get('decorators', []),
                        'complexity_score': element.complexity_score
                    })
            
            return classes
            
        except Exception as e:
            self.logger.warning(f"Failed to extract classes: {e}")
            return []
    
    def is_valid_python(self, code: str) -> bool:
        """
        Check if code is valid Python syntax.
        
        Args:
            code: Python source code
            
        Returns:
            True if code is syntactically valid
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
        except Exception:
            return False
    
    def get_code_quality_score(self, code: str) -> float:
        """
        Calculate a simple code quality score (0-100).
        
        Args:
            code: Python source code
            
        Returns:
            Quality score between 0 and 100
        """
        try:
            analysis = self.analyze_code(code)
            
            if analysis.syntax_errors:
                return 0.0
            
            metrics = analysis.metrics
            score = 100.0
            
            # Penalize low comment ratio
            comment_ratio = metrics.get('comment_ratio', 0)
            if comment_ratio < 0.1:
                score -= 20
            elif comment_ratio < 0.2:
                score -= 10
            
            # Penalize high complexity
            complexity = metrics.get('complexity_score', 0)
            code_lines = metrics.get('code_lines', 1)
            complexity_ratio = complexity / code_lines if code_lines > 0 else 0
            
            if complexity_ratio > 0.5:
                score -= 30
            elif complexity_ratio > 0.3:
                score -= 15
            
            # Penalize long lines
            max_line_length = metrics.get('max_line_length', 0)
            if max_line_length > 120:
                score -= 10
            elif max_line_length > 100:
                score -= 5
            
            # Bonus for documentation
            documented_functions = sum(
                1 for elem in analysis.elements 
                if elem.element_type in [CodeElementType.FUNCTION, CodeElementType.METHOD]
                and elem.docstring
            )
            total_functions = len(analysis.get_elements_by_type(CodeElementType.FUNCTION)) + \
                            len(analysis.get_elements_by_type(CodeElementType.METHOD))
            
            if total_functions > 0:
                doc_ratio = documented_functions / total_functions
                if doc_ratio > 0.8:
                    score += 10
                elif doc_ratio > 0.5:
                    score += 5
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate quality score: {e}")
            return 0.0 