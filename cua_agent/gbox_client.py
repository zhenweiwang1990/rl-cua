"""GBox API Client for box management and UI actions."""

import base64
import logging
from typing import Optional, Dict, Any, List, Tuple
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from cua_agent.config import GBoxConfig

logger = logging.getLogger(__name__)


class GBoxClient:
    """Client for interacting with GBox API.
    
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
        self._client = httpx.AsyncClient(
            base_url=config.api_base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
    
    async def create_box(self, box_type: Optional[str] = None) -> Dict[str, Any]:
        """Create a new GBox environment.
        
        Args:
            box_type: Type of box ("android" or "linux"), defaults to config
            
        Returns:
            Box creation response
        """
        box_type = box_type or self.config.box_type
        endpoint = f"/boxes/{box_type}"
        
        payload = {
            "wait": self.config.wait,
            "timeout": self.config.timeout,
            "config": {
                "expiresIn": self.config.expires_in,
            }
        }
        
        if self.config.labels:
            payload["config"]["labels"] = self.config.labels
        if self.config.envs:
            payload["config"]["envs"] = self.config.envs
        
        logger.info(f"Creating {box_type} box...")
        response = await self._client.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        self.box_id = result.get("id")
        logger.info(f"Box created: {self.box_id}")
        
        return result
    
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
        response = await self._client.post(
            "/boxes/terminate",
            json={"id": box_id}
        )
        response.raise_for_status()
        
        if box_id == self.box_id:
            self.box_id = None
        
        return response.json()
    
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
        
        response = await self._client.get(f"/boxes/{box_id}")
        response.raise_for_status()
        return response.json()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
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
        
        response = await self._client.post(
            f"/boxes/{box_id}/actions/screenshot",
            json={"format": format}
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Handle different response formats
        if "screenshot" in result:
            screenshot_data = result["screenshot"]
            if isinstance(screenshot_data, dict):
                # URI format
                uri = screenshot_data.get("uri", "")
                if uri.startswith("data:"):
                    # Base64 data URI
                    _, data = uri.split(",", 1)
                    image_bytes = base64.b64decode(data)
                    return image_bytes, uri
                else:
                    # HTTP URL - fetch the image
                    img_response = await self._client.get(uri)
                    img_response.raise_for_status()
                    image_bytes = img_response.content
                    base64_data = base64.b64encode(image_bytes).decode()
                    data_uri = f"data:image/{format};base64,{base64_data}"
                    return image_bytes, data_uri
            elif isinstance(screenshot_data, str):
                # Direct base64 or URL
                if screenshot_data.startswith("data:"):
                    _, data = screenshot_data.split(",", 1)
                    image_bytes = base64.b64decode(data)
                    return image_bytes, screenshot_data
                else:
                    img_response = await self._client.get(screenshot_data)
                    img_response.raise_for_status()
                    image_bytes = img_response.content
                    base64_data = base64.b64encode(image_bytes).decode()
                    data_uri = f"data:image/{format};base64,{base64_data}"
                    return image_bytes, data_uri
        
        raise ValueError(f"Unexpected screenshot response format: {result}")
    
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
        
        payload = {
            "model": self.config.model,
            "screenshot": screenshot_uri,
            "action": action,
        }
        
        response = await self._client.post("/model", json=payload)
        response.raise_for_status()
        
        return response.json()
    
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
        
        payload = {
            "x": x,
            "y": y,
            "button": button,
        }
        if double_click:
            payload["doubleClick"] = True
        
        response = await self._client.post(
            f"/boxes/{box_id}/actions/click",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
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
        
        response = await self._client.post(
            f"/boxes/{box_id}/actions/tap",
            json={"x": x, "y": y}
        )
        response.raise_for_status()
        return response.json()
    
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
        
        response = await self._client.post(
            f"/boxes/{box_id}/actions/swipe",
            json={
                "startX": start_x,
                "startY": start_y,
                "endX": end_x,
                "endY": end_y,
                "duration": duration,
            }
        )
        response.raise_for_status()
        return response.json()
    
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
        
        response = await self._client.post(
            f"/boxes/{box_id}/actions/scroll",
            json={
                "x": x,
                "y": y,
                "direction": direction,
                "distance": distance,
            }
        )
        response.raise_for_status()
        return response.json()
    
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
        
        payload = {"text": text}
        if x is not None and y is not None:
            payload["x"] = x
            payload["y"] = y
        
        response = await self._client.post(
            f"/boxes/{box_id}/actions/type",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
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
        
        response = await self._client.post(
            f"/boxes/{box_id}/actions/key",
            json={"keys": keys}
        )
        response.raise_for_status()
        return response.json()
    
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
        
        response = await self._client.post(
            f"/boxes/{box_id}/actions/button",
            json={"button": button}
        )
        response.raise_for_status()
        return response.json()
    
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
        
        response = await self._client.post(
            f"/boxes/{box_id}/actions/drag",
            json={
                "startX": start_x,
                "startY": start_y,
                "endX": end_x,
                "endY": end_y,
                "duration": duration,
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.box_id:
            try:
                await self.terminate_box()
            except Exception as e:
                logger.error(f"Failed to terminate box on exit: {e}")
        await self.close()

