"""
RoadSight AI - Geolocation & EXIF Utilities
Extracts GPS metadata from road photos or provides coordinates for map visualization.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Default fallback center coordinates (Bengaluru / India reference or customizable)
DEFAULT_LAT = 12.9716
DEFAULT_LON = 77.5946

def get_exif_gps(image_path: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract latitude and longitude from JPEG EXIF metadata."""
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if not exif_data:
                return None, None

            gps_info = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "GPSInfo":
                    for key in value:
                        sub_tag = GPSTAGS.get(key, key)
                        gps_info[sub_tag] = value[key]

            if not gps_info:
                return None, None

            def _to_deg(value):
                d = float(value[0])
                m = float(value[1])
                s = float(value[2])
                return d + (m / 60.0) + (s / 3600.0)

            lat = _to_deg(gps_info["GPSLatitude"])
            if gps_info.get("GPSLatitudeRef") == "S":
                lat = -lat

            lon = _to_deg(gps_info["GPSLongitude"])
            if gps_info.get("GPSLongitudeRef") == "W":
                lon = -lon

            return round(lat, 6), round(lon, 6)
    except Exception:
        return None, None
