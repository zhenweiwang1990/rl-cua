"""OpenTelemetry Profiler for CUA Agent training.

This module provides tracing capabilities using OpenTelemetry and Uptrace.
Each rollout is traced as a separate trace with detailed spans for:
- Box creation
- Screenshot capture
- Model inference (VLM)
- Action execution (GBox Coordinate API + GBox Action)
- Reward computation

Usage:
    from cua_agent.profiler import init_tracing, get_tracer, trace_span

    # Initialize at startup
    init_tracing(service_name="cua-agent")
    
    # Get tracer
    tracer = get_tracer("episode_runner")
    
    # Create spans
    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("key", "value")
        # ... do work ...

Environment Variables:
    UPTRACE_DSN: Uptrace DSN for exporting traces (required for Uptrace)
    OTEL_SERVICE_NAME: Service name (default: cua-agent)
    ENABLE_TRACING: Set to "true" to enable tracing (default: false)
"""

import os
import logging
import functools
from contextlib import contextmanager
from typing import Optional, Any, Callable, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

# Global state
_initialized = False
_tracer_provider = None
_enabled = False

# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, Span, Tracer
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    Span = None
    Tracer = None
    logger.warning("OpenTelemetry not installed. Tracing disabled.")

# Try to import Uptrace
try:
    import uptrace
    UPTRACE_AVAILABLE = True
except ImportError:
    UPTRACE_AVAILABLE = False
    uptrace = None
    logger.warning("Uptrace not installed. Install with: pip install uptrace")


def init_tracing(
    service_name: str = "cua-agent",
    service_version: str = "1.0.0",
    deployment_environment: str = "development",
) -> bool:
    """Initialize OpenTelemetry tracing with Uptrace.
    
    Args:
        service_name: Name of the service
        service_version: Version of the service
        deployment_environment: Deployment environment (development, production, etc.)
        
    Returns:
        True if tracing was initialized successfully, False otherwise
    """
    global _initialized, _tracer_provider, _enabled
    
    if _initialized:
        return _enabled
    
    # Check if tracing is enabled
    enable_tracing = os.getenv("ENABLE_TRACING", "false").lower() == "true"
    dsn = os.getenv("UPTRACE_DSN", "")
    
    # Print diagnostic info
    logger.info(f"[Tracing] ENABLE_TRACING={enable_tracing}, UPTRACE_DSN={'SET' if dsn else 'NOT SET'}")
    logger.info(f"[Tracing] OTEL_AVAILABLE={OTEL_AVAILABLE}, UPTRACE_AVAILABLE={UPTRACE_AVAILABLE}")
    
    if not enable_tracing:
        logger.warning("=" * 60)
        logger.warning("Tracing is DISABLED. To enable:")
        logger.warning("  export ENABLE_TRACING=true")
        logger.warning("  export UPTRACE_DSN='https://<token>@api.uptrace.dev?grpc=4317'")
        logger.warning("=" * 60)
        _initialized = True
        _enabled = False
        return False
    
    if not OTEL_AVAILABLE:
        logger.error("=" * 60)
        logger.error("OpenTelemetry not available!")
        logger.error("Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")
        logger.error("=" * 60)
        _initialized = True
        _enabled = False
        return False
    
    if not dsn:
        logger.error("=" * 60)
        logger.error("UPTRACE_DSN not set!")
        logger.error("Set with: export UPTRACE_DSN='https://<token>@api.uptrace.dev?grpc=4317'")
        logger.error("=" * 60)
        _initialized = True
        _enabled = False
        return False
    
    if not UPTRACE_AVAILABLE:
        logger.error("=" * 60)
        logger.error("Uptrace not available!")
        logger.error("Install with: pip install uptrace")
        logger.error("=" * 60)
        _initialized = True
        _enabled = False
        return False
    
    try:
        # Configure Uptrace
        uptrace.configure_opentelemetry(
            dsn=dsn,
            service_name=service_name,
            service_version=service_version,
            deployment_environment=deployment_environment,
            resource_attributes={
                "service.namespace": "cua-training",
            }
        )
        
        _initialized = True
        _enabled = True
        logger.info("=" * 60)
        logger.info(f"✅ Tracing initialized successfully!")
        logger.info(f"   Service: {service_name}")
        logger.info(f"   Version: {service_version}")
        logger.info(f"   Environment: {deployment_environment}")
        logger.info(f"   DSN: {dsn[:50]}...")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ Failed to initialize tracing: {e}")
        logger.error("   Check your UPTRACE_DSN and network connectivity")
        logger.error("=" * 60, exc_info=True)
        _initialized = True
        _enabled = False
        return False


def shutdown_tracing():
    """Shutdown tracing and flush all pending spans."""
    global _initialized, _enabled
    
    if _enabled and UPTRACE_AVAILABLE and uptrace:
        try:
            uptrace.shutdown()
            logger.info("Tracing shutdown complete.")
        except Exception as e:
            logger.warning(f"Error during tracing shutdown: {e}")
    
    _initialized = False
    _enabled = False


def get_tracer(name: str = "cua_agent") -> "Tracer":
    """Get a tracer instance.
    
    Args:
        name: Name of the tracer (usually module name)
        
    Returns:
        Tracer instance (or NoOpTracer if tracing is disabled)
    """
    if not _enabled or not OTEL_AVAILABLE:
        return NoOpTracer()
    
    return trace.get_tracer(name, "1.0.0")


class NoOpSpan:
    """No-op span for when tracing is disabled."""
    
    def set_attribute(self, key: str, value: Any) -> None:
        pass
    
    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        pass
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def set_status(self, status: Any, description: Optional[str] = None) -> None:
        pass
    
    def record_exception(self, exception: Exception) -> None:
        pass
    
    def end(self, end_time: Optional[int] = None) -> None:
        pass
    
    def is_recording(self) -> bool:
        return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class NoOpTracer:
    """No-op tracer for when tracing is disabled."""
    
    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        kind: Any = None,
        attributes: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        yield NoOpSpan()
    
    def start_span(
        self,
        name: str,
        kind: Any = None,
        attributes: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> NoOpSpan:
        return NoOpSpan()


@contextmanager
def trace_span(
    name: str,
    tracer_name: str = "cua_agent",
    attributes: Optional[Dict[str, Any]] = None,
    kind: Any = None,
):
    """Context manager for creating a traced span.
    
    Args:
        name: Name of the span
        tracer_name: Name of the tracer
        attributes: Attributes to set on the span
        kind: Span kind (e.g., SpanKind.CLIENT, SpanKind.SERVER)
        
    Yields:
        The span object
        
    Example:
        with trace_span("my_operation", attributes={"key": "value"}) as span:
            # ... do work ...
            span.set_attribute("result", "success")
    """
    tracer = get_tracer(tracer_name)
    
    kwargs = {}
    if kind is not None:
        kwargs["kind"] = kind
    if attributes:
        kwargs["attributes"] = attributes
    
    with tracer.start_as_current_span(name, **kwargs) as span:
        try:
            yield span
        except Exception as e:
            if _enabled and OTEL_AVAILABLE:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            raise


def trace_function(
    name: Optional[str] = None,
    tracer_name: str = "cua_agent",
    record_args: bool = False,
    record_result: bool = False,
):
    """Decorator for tracing a function.
    
    Args:
        name: Span name (default: function name)
        tracer_name: Name of the tracer
        record_args: Whether to record function arguments as attributes
        record_result: Whether to record function result as attribute
        
    Example:
        @trace_function(record_args=True)
        async def my_function(arg1, arg2):
            return result
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer(tracer_name)
            
            with tracer.start_as_current_span(span_name) as span:
                if record_args and _enabled:
                    # Record positional args (skip self)
                    for i, arg in enumerate(args[1:] if args else []):
                        span.set_attribute(f"arg_{i}", str(arg)[:100])
                    # Record keyword args
                    for key, value in kwargs.items():
                        span.set_attribute(f"kwarg_{key}", str(value)[:100])
                
                try:
                    result = await func(*args, **kwargs)
                    
                    if record_result and _enabled and result is not None:
                        span.set_attribute("result", str(result)[:200])
                    
                    return result
                    
                except Exception as e:
                    if _enabled and OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer(tracer_name)
            
            with tracer.start_as_current_span(span_name) as span:
                if record_args and _enabled:
                    for i, arg in enumerate(args[1:] if args else []):
                        span.set_attribute(f"arg_{i}", str(arg)[:100])
                    for key, value in kwargs.items():
                        span.set_attribute(f"kwarg_{key}", str(value)[:100])
                
                try:
                    result = func(*args, **kwargs)
                    
                    if record_result and _enabled and result is not None:
                        span.set_attribute("result", str(result)[:200])
                    
                    return result
                    
                except Exception as e:
                    if _enabled and OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def get_trace_url(span: Any) -> Optional[str]:
    """Get the Uptrace URL for a span.
    
    Args:
        span: The span to get the URL for
        
    Returns:
        Uptrace URL or None if not available
    """
    if not _enabled or not UPTRACE_AVAILABLE or not uptrace:
        return None
    
    try:
        return uptrace.trace_url(span)
    except Exception:
        return None


# Convenience exports
__all__ = [
    "init_tracing",
    "shutdown_tracing",
    "get_tracer",
    "trace_span",
    "trace_function",
    "get_trace_url",
    "NoOpSpan",
    "NoOpTracer",
]

