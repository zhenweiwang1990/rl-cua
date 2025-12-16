"""GBox API Client for box management and UI actions using official SDK."""

import base64
import binascii
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from gbox_sdk import GboxSDK

from cua_agent.config import GBoxConfig

# Type alias for SDK response
ResponseDict = Dict[str, Any]

logger = logging.getLogger(__name__)


class GBoxClient:
    """Client for interacting with GBox API using official SDK.
    
    Handles:
    - Box creation and termination
    - Screenshot capture
    - UI actions (click, swipe, scroll, type, key press, button press)
    - Coordinate generation using gbox-handy-1 model
    """
    
    def __init__(self, config: GBoxConfig):
        """Initialize GBox client.
        
        Args:
            config: GBox configuration
        """
        self.config = config
        self.box_id: Optional[str] = None
        self._sdk = GboxSDK(api_key=config.api_key)
        self._box: Optional[Any] = None  # Box resource object
    
    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse SDK response to dictionary.
        
        Args:
            response: SDK response object
            
        Returns:
            Dictionary representation of response
        """
        if hasattr(response, 'json'):
            return response.json()
        elif hasattr(response, 'data'):
            if hasattr(response.data, 'model_dump'):
                return response.data.model_dump()
            elif hasattr(response.data, 'dict'):
                return response.data.dict()
            else:
                return dict(response.data) if hasattr(response.data, '__dict__') else response.data
        elif hasattr(response, 'model_dump'):
            return response.model_dump()
        elif hasattr(response, 'dict'):
            return response.dict()
        else:
            return dict(response) if isinstance(response, dict) else {"response": response}
    
    async def create_box(self, box_type: Optional[str] = None) -> Dict[str, Any]:
        """Create a new GBox environment.
        
        Args:
            box_type: Type of box ("android" or "linux"), defaults to config
            
        Returns:
            Box creation response
        """
        box_type = box_type or self.config.box_type
        logger.info(f"Creating {box_type} box...")
        
        try:
            # Create box using SDK - according to official docs
            # https://docs.gbox.ai/api-reference/box/create-android-box
            # Note: SDK create() is synchronous, not async
            box_response = self._sdk.create(
                type=box_type,
                wait=self.config.wait,
                timeout=self.config.timeout,
                config={
                    "expiresIn": self.config.expires_in,
                    **({"labels": self.config.labels} if self.config.labels else {}),
                    **({"envs": self.config.envs} if self.config.envs else {}),
                }
            )
            
            # Store box response
            self._box = box_response
            # Get box ID from response.data.id (according to official example)
            if hasattr(box_response, 'data') and hasattr(box_response.data, 'id'):
                self.box_id = box_response.data.id
            elif hasattr(box_response, 'id'):
                self.box_id = box_response.id
            elif isinstance(box_response, dict):
                self.box_id = box_response.get("id") or (box_response.get("data", {}).get("id") if isinstance(box_response.get("data"), dict) else None)
            else:
                raise ValueError(f"Unexpected box response format: {type(box_response)}")
            
            if not self.box_id:
                raise ValueError("Failed to extract box ID from response")
            
            logger.info(f"Box created: {self.box_id}")
            return {"id": self.box_id}
            
        except Exception as e:
            logger.error(f"Failed to create {box_type} box: {e}")
            raise RuntimeError(
                f"Cannot create GBox. Please check:\n"
                f"  1. API key is valid\n"
                f"  2. Network connectivity\n"
                f"  3. Box type is supported: {box_type}"
            ) from e
    
    async def terminate_box(self, box_id: Optional[str] = None) -> Dict[str, Any]:
        """Terminate a GBox environment.
        
        Args:
            box_id: Box ID to terminate, defaults to current box
            
        Returns:
            Termination response
        """
        box_id = box_id or self.box_id
        if not box_id:
            raise ValueError("No box ID provided")
        
        logger.info(f"Terminating box: {box_id}")
        
        try:
            # Use SDK terminate method if available, otherwise use DELETE endpoint
            if hasattr(self._sdk, 'terminate'):
                result = self._sdk.terminate(box_id)
            else:
                # Use DELETE endpoint: DELETE /boxes/{boxId}
                result = self._sdk.client.delete(
                    f"/boxes/{box_id}",
                    cast_to=ResponseDict
                )
            return self._parse_response(result) if result else {"id": box_id, "status": "terminated"}
            
            if box_id == self.box_id:
                self.box_id = None
                self._box = None
            
            return {"id": box_id, "status": "terminated"}
        except Exception as e:
            logger.error(f"Failed to terminate box: {e}")
            raise
    
    async def get_box(self, box_id: Optional[str] = None) -> Dict[str, Any]:
        """Get box information.
        
        Args:
            box_id: Box ID, defaults to current box
            
        Returns:
            Box information
        """
        box_id = box_id or self.box_id
        if not box_id:
            raise ValueError("No box ID provided")
        
        try:
            result = self._sdk.client.get(f"/boxes/{box_id}", cast_to=ResponseDict)
            return self._parse_response(result)
        except Exception as e:
            logger.error(f"Failed to get box info: {e}")
            raise
    
    async def take_screenshot(
        self,
        box_id: Optional[str] = None,
        format: str = "png",
    ) -> Tuple[bytes, str]:
        """Take a screenshot of the box display.
        
        Args:
            box_id: Box ID, defaults to current box
            format: Image format ("png" or "jpeg")
            
        Returns:
            Tuple of (image_bytes, base64_data_uri)
        """
        box_id = box_id or self.box_id
        if not box_id:
            raise ValueError("No box ID provided")
        
        try:
            # Use SDK client to call screenshot API
            # According to docs: POST /boxes/{boxId}/actions/screenshot
            # cast_to is required - use dict as default response type
            # body should be dict, SDK will serialize it
            screenshot_result = self._sdk.client.post(
                f"/boxes/{box_id}/actions/screenshot",
                cast_to=ResponseDict,
                body={"format": format}
            )
            
            # Handle screenshot result - SDK returns dict directly when cast_to=ResponseDict
            result_data = screenshot_result if isinstance(screenshot_result, dict) else self._parse_response(screenshot_result)
            
            # Log response for debugging (use INFO level since DEBUG might not be enabled)
            logger.info(f"Screenshot response type: {type(result_data)}, keys: {list(result_data.keys()) if isinstance(result_data, dict) else 'N/A'}")
            
            # Extract screenshot data from response
            # Response format: {"screenshot": "data:image/png;base64,..."} or {"uri": "https://..."}
            screenshot_data = None
            if isinstance(result_data, dict):
                # Try different possible keys (uri is most common based on logs)
                screenshot_data = (
                    result_data.get("uri") or  # Most common format
                    result_data.get("screenshot") or 
                    result_data.get("url") or
                    (result_data.get("data", {}).get("uri") if isinstance(result_data.get("data"), dict) else None) or
                    (result_data.get("data", {}).get("screenshot") if isinstance(result_data.get("data"), dict) else None)
                )
            
            if not screenshot_data or (isinstance(screenshot_data, str) and not screenshot_data.strip()):
                logger.error(f"Unexpected screenshot response format: {result_data}")
                logger.error(f"Available keys: {list(result_data.keys()) if isinstance(result_data, dict) else 'N/A'}")
                raise ValueError(f"Failed to extract screenshot from response: {result_data}")
            
            # Convert to bytes and data URI
            if isinstance(screenshot_data, str):
                if screenshot_data.startswith("data:"):
                    # Base64 data URI
                    try:
                        parts = screenshot_data.split(",", 1)
                        if len(parts) != 2:
                            raise ValueError(f"Invalid data URI format: {screenshot_data[:50]}...")
                        _, data = parts
                        image_bytes = base64.b64decode(data)
                        return image_bytes, screenshot_data
                    except (ValueError, base64.binascii.Error) as e:
                        logger.error(f"Failed to parse data URI: {e}, data length: {len(screenshot_data)}")
                        raise ValueError(f"Invalid data URI format: {screenshot_data[:50]}...")
                else:
                    # HTTP URL - fetch the image
                    import httpx
                    async with httpx.AsyncClient() as client:
                        img_response = await client.get(screenshot_data)
                        img_response.raise_for_status()
                        image_bytes = img_response.content
                        base64_data = base64.b64encode(image_bytes).decode()
                        data_uri = f"data:image/{format};base64,{base64_data}"
                        return image_bytes, data_uri
            
            raise ValueError(f"Unexpected screenshot data type: {type(screenshot_data)}, value: {str(screenshot_data)[:100]}")
            
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}", exc_info=True)
            raise
    
    async def generate_coordinates(
        self,
        screenshot_uri: str,
        action_type: str,
        target: str,
        end_target: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate coordinates using gbox-handy-1 model.
        
        Args:
            screenshot_uri: Screenshot as base64 data URI or HTTP URL
            action_type: Type of action ("click", "drag", "scroll")
            target: Target element description
            end_target: End target for drag actions
            direction: Direction for scroll actions
            
        Returns:
            Coordinate generation response
        """
        # Build action object based on type
        if action_type == "click":
            action = {
                "type": "click",
                "target": target,
            }
        elif action_type == "drag":
            action = {
                "type": "drag",
                "startTarget": target,
                "endTarget": end_target or target,
            }
        elif action_type == "scroll":
            action = {
                "type": "scroll",
                "target": target,
                "direction": direction or "down",
            }
        else:
            raise ValueError(f"Unknown action type: {action_type}")
        
        try:
            # Use SDK client to call model API
            # According to docs: POST /model
            result = self._sdk.client.post(
                "/model",
                cast_to=ResponseDict,
                body={
                    "model": self.config.model,
                    "screenshot": screenshot_uri,
                    "action": action,
                }
            )
            
            return self._parse_response(result)
                
        except Exception as e:
            logger.error(f"Failed to generate coordinates: {e}")
            raise
    
    async def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        double_click: bool = False,
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform a click action.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Button type ("left", "right", "middle")
            double_click: Whether to double click
            box_id: Box ID, defaults to current box
            
        Returns:
            Action response
        """
        box_id = box_id or self.box_id
        if not box_id:
            raise ValueError("No box ID provided")
        
        try:
            payload = {"x": x, "y": y, "button": button}
            if double_click:
                payload["doubleClick"] = True
            
            result = self._sdk.client.post(
                f"/boxes/{box_id}/actions/click",
                cast_to=ResponseDict,
                body=payload
            )
            
            return self._parse_response(result)
        except Exception as e:
            logger.error(f"Failed to click: {e}")
            raise
    
    async def tap(
        self,
        x: int,
        y: int,
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform a tap action (for touch screens).
        
        Args:
            x: X coordinate
            y: Y coordinate
            box_id: Box ID, defaults to current box
            
        Returns:
            Action response
        """
        box_id = box_id or self.box_id
        if not box_id:
            raise ValueError("No box ID provided")
        
        try:
            result = self._sdk.client.post(
                f"/boxes/{box_id}/actions/tap",
                cast_to=ResponseDict,
                body={"x": x, "y": y}
            )
            
            return self._parse_response(result)
        except Exception as e:
            logger.error(f"Failed to tap: {e}")
            raise
    
    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: int = 300,
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform a swipe action.
        
        Args:
            start_x: Start X coordinate
            start_y: Start Y coordinate
            end_x: End X coordinate
            end_y: End Y coordinate
            duration: Swipe duration in ms
            box_id: Box ID, defaults to current box
            
        Returns:
            Action response
        """
        box_id = box_id or self.box_id
        if not box_id:
            raise ValueError("No box ID provided")
        
        try:
            result = self._sdk.client.post(
                f"/boxes/{box_id}/actions/swipe",
                cast_to=ResponseDict,
                body={
                    "startX": start_x,
                    "startY": start_y,
                    "endX": end_x,
                    "endY": end_y,
                    "duration": duration,
                }
            )
            
            return self._parse_response(result)
        except Exception as e:
            logger.error(f"Failed to swipe: {e}")
            raise
    
    async def scroll(
        self,
        x: int,
        y: int,
        direction: str = "down",
        distance: int = 300,
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform a scroll action.
        
        Args:
            x: X coordinate of scroll center
            y: Y coordinate of scroll center
            direction: Scroll direction ("up", "down", "left", "right")
            distance: Scroll distance in pixels
            box_id: Box ID, defaults to current box
            
        Returns:
            Action response
        """
        box_id = box_id or self.box_id
        if not box_id:
            raise ValueError("No box ID provided")
        
        try:
            result = self._sdk.client.post(
                f"/boxes/{box_id}/actions/scroll",
                cast_to=ResponseDict,
                body={
                    "x": x,
                    "y": y,
                    "direction": direction,
                    "distance": distance,
                }
            )
            
            return self._parse_response(result)
        except Exception as e:
            logger.error(f"Failed to scroll: {e}")
            raise
    
    async def type_text(
        self,
        text: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Type text into the focused element or at coordinates.
        
        Args:
            text: Text to type
            x: Optional X coordinate to click first
            y: Optional Y coordinate to click first
            box_id: Box ID, defaults to current box
            
        Returns:
            Action response
        """
        box_id = box_id or self.box_id
        if not box_id:
            raise ValueError("No box ID provided")
        
        try:
            params = {"text": text}
            if x is not None and y is not None:
                params["x"] = x
                params["y"] = y
            result = self._sdk.client.post(
                f"/boxes/{box_id}/actions/type",
                cast_to=ResponseDict,
                body=params
            )
            
            return self._parse_response(result)
        except Exception as e:
            logger.error(f"Failed to type text: {e}")
            raise
    
    async def press_key(
        self,
        keys: List[str],
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Press one or more keys.
        
        Args:
            keys: List of key names to press (e.g., ["Enter"], ["Ctrl", "A"])
            box_id: Box ID, defaults to current box
            
        Returns:
            Action response
        """
        box_id = box_id or self.box_id
        if not box_id:
            raise ValueError("No box ID provided")
        
        try:
            result = self._sdk.client.post(
                f"/boxes/{box_id}/actions/key",
                cast_to=ResponseDict,
                body={"keys": keys}
            )
            
            return self._parse_response(result)
        except Exception as e:
            logger.error(f"Failed to press key: {e}")
            raise
    
    async def press_button(
        self,
        button: str,
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Press a device button (Android).
        
        Args:
            button: Button name ("back", "home", "menu")
            box_id: Box ID, defaults to current box
            
        Returns:
            Action response
        """
        box_id = box_id or self.box_id
        if not box_id:
            raise ValueError("No box ID provided")
        
        try:
            result = self._sdk.client.post(
                f"/boxes/{box_id}/actions/button",
                cast_to=ResponseDict,
                body={"button": button}
            )
            
            return self._parse_response(result)
        except Exception as e:
            logger.error(f"Failed to press button: {e}")
            raise
    
    async def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: int = 500,
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform a drag action.
        
        Args:
            start_x: Start X coordinate
            start_y: Start Y coordinate
            end_x: End X coordinate
            end_y: End Y coordinate
            duration: Drag duration in ms
            box_id: Box ID, defaults to current box
            
        Returns:
            Action response
        """
        box_id = box_id or self.box_id
        if not box_id:
            raise ValueError("No box ID provided")
        
        try:
            result = self._sdk.client.post(
                f"/boxes/{box_id}/actions/drag",
                cast_to=ResponseDict,
                body={
                    "startX": start_x,
                    "startY": start_y,
                    "endX": end_x,
                    "endY": end_y,
                    "duration": duration,
                }
            )
            
            return self._parse_response(result)
        except Exception as e:
            logger.error(f"Failed to drag: {e}")
            raise
    
    async def close(self):
        """Close the SDK client."""
        # SDK handles cleanup automatically
        if self._box:
            try:
                await self.terminate_box()
            except Exception as e:
                logger.warning(f"Failed to terminate box on close: {e}")
        self._box = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.box_id:
            try:
                await self.terminate_box()
            except Exception as e:
                logger.error(f"Failed to terminate box on exit: {e}")
        await self.close()
