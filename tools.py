from util import get_logger
from context import get_dataset_path, set_dataset_path

import requests
import trafilatura
import piexif
import haversine

import base64
import os
import io

logger = get_logger(__name__)

def get_exif_data(image_path: str) -> dict:
    image_path = os.path.join(get_dataset_path(), "images", image_path)
    logger.debug(f"Extracting EXIF data from: {image_path}")

    try:
        exif_dict = piexif.load(image_path, True)
        
        if 'thumbnail' in exif_dict:
            del exif_dict['thumbnail']

        logger.debug(f"Successfully extracted EXIF data with {len(exif_dict)} sections")
        return exif_dict
    except Exception as e:
        if "No EXIF data found" in str(e) or "Given file is neither JPEG nor TIFF" in str(e):
            logger.function_call(f"No EXIF data found for {image_path}")
            return None
        else:
            logger.error(f"Error extracting EXIF data from {image_path}: {str(e)}")
            return None

def google_web_search(query: str, limit: int = 10) -> list:
    import requests
    from bs4 import BeautifulSoup
    from googlesearch import search
    
    try:
        results = []
        for i, url in enumerate(search(query, num_results=limit, sleep_interval=1)):
            title = f"Result {i+1}"  # Fallback

            try:
                response = requests.get(url, timeout=5, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                soup = BeautifulSoup(response.text, 'html.parser')
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.get_text().strip()[:200]  # Limit length
            except:
                pass  # Keep fallback title
            
            results.append({
                'title': title,
                'url': url,
                'description': ''
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]

def visit_website(url: str) -> dict:
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        # Extract main content
        content = trafilatura.extract(response.text, include_comments=False)
        title = trafilatura.extract_metadata(response.text).title if trafilatura.extract_metadata(response.text) else "No title"
        
        if not content:
            content = "Could not extract main content"
        
        return {
            "url": response.url,
            "title": title,
            "description": "",
            "content": content[:8000] + ("... [truncated]" if len(content) > 8000 else ""),
            "status_code": response.status_code,
            "content_type": response.headers.get('content-type', ''),
            "content_length": len(content)
        }
    except Exception as e:
        return {"error": f"Failed to process website: {str(e)}"}

def overpass_turbo_query(query: str, timeout: int = 60) -> dict:
    """
    Execute an Overpass QL query against OpenStreetMap data
    
    Args:
        query: Overpass QL query string
        timeout: Query timeout in seconds (default: 25)
    
    Returns:
        Dictionary with query results or error information
    """
    import requests
    import json
    
    logger.debug(f"Executing Overpass query: {query}...")
    
    # Validate inputs
    if not query or not isinstance(query, str):
        logger.error("Invalid query parameter for Overpass")
        return {"error": "Invalid query parameter"}
    
    # Overpass API endpoint
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Add timeout to query if not already present
    if not query.strip().startswith('['):
        query = f"[out:json][timeout:{timeout}];\n{query}"
    elif 'timeout:' not in query:
        # Insert timeout into existing settings
        query = query.replace('[out:json]', f'[out:json][timeout:{timeout}]')
    
    try:
        # Make the request
        response = requests.post(
            overpass_url,
            data=query,
            headers={
                'Content-Type': 'text/plain; charset=utf-8',
                'User-Agent': 'OSINTbench'
            },
            timeout=timeout + 5  # Add buffer to request timeout
        )
        response.raise_for_status()
        
        # Parse JSON response
        result_data = response.json()
        
        # Extract elements if present
        elements = result_data.get('elements', [])
        
        # Process and summarize results
        summary = {
            "total_elements": len(elements),
            "element_types": {},
            "has_coordinates": 0,
            "has_tags": 0
        }
        
        # Analyze elements
        for element in elements:
            elem_type = element.get('type', 'unknown')
            summary["element_types"][elem_type] = summary["element_types"].get(elem_type, 0) + 1
            
            if 'lat' in element and 'lon' in element:
                summary["has_coordinates"] += 1
            
            if element.get('tags'):
                summary["has_tags"] += 1
        
        # Limit response size for large datasets
        if len(elements) > 100:
            logger.debug(f"Large result set ({len(elements)} elements), truncating to first 100")
            elements = elements[:100]
            summary["truncated"] = True
            summary["original_count"] = len(result_data.get('elements', []))
        
        result = {
            "success": True,
            "query": query,
            "summary": summary,
            "elements": elements,
            "generator": result_data.get('generator', 'unknown'),
            "osm3s": result_data.get('osm3s', {})
        }
        
        logger.debug(f"Overpass query successful: {summary['total_elements']} elements found")
        return result
        
    except requests.exceptions.Timeout:
        logger.error(f"Overpass query timed out after {timeout}s")
        return {"error": f"Query timed out after {timeout} seconds"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Overpass request failed: {str(e)}")
        return {"error": f"Request failed: {str(e)}"}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from Overpass: {str(e)}")
        return {"error": f"Invalid JSON response: {str(e)}"}
    except Exception as e:
        logger.error(f"Overpass query error: {str(e)}")
        return {"error": f"Query error: {str(e)}"}

def geocode(query: str, limit: int = 5) -> dict:
    """
    Geocode a location using OpenStreetMap's Nominatim service
    
    Args:
        query: Location to search for (e.g., "Niagara Falls", "123 Main St, Toronto")
        limit: Maximum number of results to return (default: 5)
    
    Returns:
        Dictionary with geocoding results or error information
    """
    import requests
    import json
    
    logger.debug(f"Geocoding query: {query} (limit: {limit})")
    
    # Validate inputs
    if not query or not isinstance(query, str):
        logger.error("Invalid query parameter for Nominatim")
        return {"error": "Invalid query parameter"}
    
    if limit < 1 or limit > 50:
        limit = 5
        logger.warning("Limit adjusted to 5 (valid range: 1-50)")
    
    # Nominatim API endpoint
    nominatim_url = "https://nominatim.openstreetmap.org/search"
    
    # Request parameters
    params = {
        'q': query,
        'format': 'json',
        'limit': limit,
        'addressdetails': 1,
        'extratags': 1,
        'namedetails': 1
    }
    
    # Headers to identify the request
    headers = {
        'User-Agent': 'OSINTbench/1.0',
        'Accept': 'application/json'
    }
    
    try:
        # Make the request
        response = requests.get(
            nominatim_url,
            params=params,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        
        # Parse JSON response
        results = response.json()
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            processed_result = {
                "place_id": result.get("place_id"),
                "display_name": result.get("display_name"),
                "latitude": float(result.get("lat", 0)),
                "longitude": float(result.get("lon", 0)),
                "importance": result.get("importance", 0),
                "place_rank": result.get("place_rank"),
                "category": result.get("category"),
                "type": result.get("type"),
                "address": result.get("address", {}),
                "boundingbox": result.get("boundingbox", []),
                "extratags": result.get("extratags", {}),
                "namedetails": result.get("namedetails", {})
            }
            processed_results.append(processed_result)
        
        # Create summary
        summary = {
            "query": query,
            "total_results": len(results),
            "countries": list(set(r.get("address", {}).get("country") for r in results if r.get("address", {}).get("country"))),
            "types": list(set(r.get("type") for r in results if r.get("type"))),
            "categories": list(set(r.get("category") for r in results if r.get("category")))
        }
        
        result_dict = {
            "success": True,
            "summary": summary,
            "results": processed_results
        }
        
        logger.debug(f"Nominatim geocoding successful: {len(results)} results for '{query}'")
        return result_dict
        
    except requests.exceptions.Timeout:
        logger.error("Nominatim request timed out")
        return {"error": "Request timed out after 10 seconds"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Nominatim request failed: {str(e)}")
        return {"error": f"Request failed: {str(e)}"}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from Nominatim: {str(e)}")
        return {"error": f"Invalid JSON response: {str(e)}"}
    except Exception as e:
        logger.error(f"Nominatim geocoding error: {str(e)}")
        return {"error": f"Geocoding error: {str(e)}"}

def sv_query(lat: float, lng: float, zoom: int = 4) -> dict:
    """
    Get Street View panorama image for given coordinates
    
    Args:
        lat (float): Latitude coordinate
        lng (float): Longitude coordinate  
        zoom (int): Zoom level (0-5, higher = more detail, default: 4)
        
    Returns:
        dict: Dictionary containing success status and base64 image data
    """
    from streetview import search_panoramas, get_panorama
    from PIL import Image
    
    logger.function_call(f"Getting Street View panorama for coordinates: {lat}, {lng}")
    
    try:
        panoramas = search_panoramas(lat, lng)
        
        if not panoramas:
            logger.error(f"No Street View panoramas found at coordinates: {lat}, {lng}")
            return {
                "success": False,
                "error": f"No Street View imagery available at coordinates: {lat}, {lng}",
                "coordinates": {"lat": lat, "lng": lng}
            }
        
        first = panoramas[-1]
        distance_from_input = haversine.haversine((lat, lng), (first.lat, first.lon))
        
        zoom = max(0, min(5, zoom))
        
        panorama = get_panorama(first.pano_id, zoom=zoom, multi_threaded=True)
        original_size = panorama.size
        
        # Downscale to reasonable size for LLM context usage
        # Street View panoramas are typically 2:1 ratio, so use 1024x512 max
        MAX_WIDTH = 2048
        MAX_HEIGHT = 1024
        
        if panorama.size[0] > MAX_WIDTH or panorama.size[1] > MAX_HEIGHT:
            # Calculate scale factor to fit within bounds while maintaining aspect ratio
            width_scale = MAX_WIDTH / panorama.size[0]
            height_scale = MAX_HEIGHT / panorama.size[1]
            scale_factor = min(width_scale, height_scale)
            
            new_width = int(panorama.size[0] * scale_factor)
            new_height = int(panorama.size[1] * scale_factor)
            
            panorama = panorama.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.debug(f"Resized panorama from {original_size} to {panorama.size} (scale: {scale_factor:.3f})")
        
        # Save full resolution version for reference
        save_path = os.path.join(get_dataset_path(), "street_view_cache", f"panorama_{lat}_{lng}_z{zoom}.jpg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the original full-res version
        full_panorama = get_panorama(first.pano_id, zoom=zoom, multi_threaded=True)
        full_panorama.save(save_path, format='JPEG', quality=85, optimize=True)
        
        panorama.save('panorama.jpg', quality=40, optimize=True)
        
        # Process the downscaled version for base64
        img_buffer = io.BytesIO()
        panorama.save(img_buffer, format='JPEG', quality=40, optimize=True)  # Lower quality for smaller size
        img_buffer.seek(0)
        
        # Check final size and estimate tokens
        img_size = len(img_buffer.getvalue())
        estimated_tokens = (img_size * 4/3) * 0.75  # base64 expansion * token ratio
        
        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        result = {
            "success": True,
            "image_data": img_base64,
            "coordinates": {"lat": first.lat, "lng": first.lon},
            "distance_from_input": str(distance_from_input) + " km"
        }
        
        logger.function_call(f"Successfully retrieved Street View panorama: {panorama.size[0]}x{panorama.size[1]} pixels (~{estimated_tokens:,.0f} tokens)")
        return result
    except Exception as e:
        logger.error(f"Error retrieving Street View panorama: {str(e)}")
        return {
            "success": False,
            "error": f"Error retrieving Street View panorama: {str(e)}"
        }

def view_image_from_reverse_image_search(case_image_id: int, result_image_id: int) -> dict:
    import json
    import re
    from PIL import Image

    cache_path = os.path.join(get_dataset_path(), "reverse-image-cache", f"{case_image_id}.txt")
    logger.debug(f"Viewing image {result_image_id} from reverse search cache: {cache_path}")

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            if result_image_id >= len(data):
                logger.warning(f"Image ID {result_image_id} not found in cache (max: {len(data)-1})")
                return {
                    "id": result_image_id,
                    "success": False,
                    "error": f"Image ID {result_image_id} not found. Available IDs: 0-{len(data)-1}"
                }

            image_data = data[result_image_id]
            img_b64 = image_data['img']
            
            # Handle different base64 formats
            if img_b64.startswith('data:image/'):
                # Extract the actual base64 data from data URL
                # Format: data:image/jpeg;base64,/9j/4AAQ...
                match = re.match(r'data:image/([^;]+);base64,(.+)', img_b64)
                if match:
                    format_type = match.group(1).upper()
                    if format_type == 'JPG':
                        format_type = 'JPEG'
                    actual_b64 = match.group(2)
                    logger.debug(f"Extracted {format_type} image from data URL")
                else:
                    logger.error("Invalid data URL format in cached image")
                    return {
                        "id": result_image_id,
                        "success": False,
                        "error": "Invalid data URL format"
                    }
            else:
                # Assume it's just raw base64 data
                actual_b64 = img_b64
                format_type = image_data.get('format', 'JPEG').upper()
                logger.debug(f"Using raw base64 data, format: {format_type}")

            # Validate base64 and get image info
            try:
                img_bytes = base64.b64decode(actual_b64)
                img = Image.open(io.BytesIO(img_bytes))
                
                logger.debug(f"Successfully decoded image: {img.size} {format_type}")
                return {
                    "id": result_image_id,
                    "success": True,
                    "found_image": actual_b64,  # Return clean base64 without data URL prefix
                    "image_info": {
                        "size": img.size,
                        "format": format_type,
                        "source_url": image_data.get('url', 'Unknown'),
                        "source_title": image_data.get('title', 'Unknown')
                    }
                }
            except Exception as decode_error:
                logger.error(f"Failed to decode base64 image {result_image_id}: {str(decode_error)}")
                return {
                    "id": result_image_id,
                    "success": False,
                    "error": f"Failed to decode base64 image: {str(decode_error)}"
                }

    except FileNotFoundError:
        logger.error(f"Cache file not found: {cache_path}")
        return {
            "id": result_image_id,
            "success": False,
            "error": f"Cache file not found: {cache_path}"
        }
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in cache file {cache_path}: {str(e)}")
        return {
            "id": result_image_id,
            "success": False,
            "error": f"Invalid JSON in cache file: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Error loading cache for image {result_image_id}: {str(e)}")
        return {
            "id": result_image_id,
            "success": False,
            "error": f"Error loading cache: {str(e)}"
        }

def reverse_image_search(image_path: str, use_cache: bool = True) -> list:
    """
    Performs a reverse image search using Google Images and returns the first 10 results as a list of dictionaries.
    
    Args:
        image_path (str): Path to the image file (e.g., "dataset/basic/images/16.jpg")
        use_cache (bool): Whether to use cached results if available
        
    Returns:
        list: List of dictionaries containing search results with 'title', 'url', 'source' keys
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import time
    import random
    import json

    image_path = os.path.join(get_dataset_path(), "images", image_path)
    logger.function_call(f"Starting reverse image search for: {image_path}")

    def get_cache_path(image_path: str) -> str:
        """Generate cache file path based on image path"""
        path_parts = image_path.replace('\\', '/').split('/')
        
        if len(path_parts) >= 3 and path_parts[-2] == 'images':
            # Extract dataset name and image filename
            dataset_parts = path_parts[:-2]  # Everything except 'images' and filename
            image_filename = path_parts[-1]  # Just the filename
            
            # Remove extension and add .txt
            cache_filename = os.path.splitext(image_filename)[0] + '.txt'
            
            # Build cache directory path
            cache_dir = os.path.join(*dataset_parts, 'reverse-image-cache')
            cache_path = os.path.join(cache_dir, cache_filename)
            
            return cache_path
        else:
            # Fallback: create cache next to the image file
            image_dir = os.path.dirname(image_path)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            cache_dir = os.path.join(image_dir, 'reverse-image-cache')
            return os.path.join(cache_dir, f"{image_name}.txt")
    
    def load_cache(cache_path: str) -> list:
        """Load cached results if they exist"""
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    logger.function_call(f"Loaded {len(cached_data)} results from cache: {cache_path}")
                    return cached_data
        except Exception as e:
            logger.warning(f"Could not load cache from {cache_path}: {e}")
        return None
    
    def save_cache(cache_path: str, results: list):
        """Save results to cache"""
        try:
            # Create cache directory if it doesn't exist
            cache_dir = os.path.dirname(cache_path)
            os.makedirs(cache_dir, exist_ok=True)
            
            # Save results as JSON
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.function_call(f"Saved {len(results)} results to cache: {cache_path}")
            
        except Exception as e:
            logger.warning(f"Could not save cache to {cache_path}: {e}")
    
    # Check if image file exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found at {image_path}")
        return []
    
    # Get cache path and check for cached results
    cache_path = get_cache_path(image_path)
    
    if use_cache:
        cached_results = load_cache(cache_path)
        if cached_results:
            logger.debug("Processing cached results...")
            # Create new list without 'img' fields instead of deleting
            clean_results = []
            for i, result in enumerate(cached_results):
                logger.debug(f"Processing cached result {i+1}/{len(cached_results)}")
                clean_result = {k: v for k, v in result.items() if k != 'img'}
                clean_results.append(clean_result)
            
            logger.debug("Finished processing cached results")
            return clean_results
        
    try:
        # Get absolute path for the image
        image_path_abs = os.path.abspath(image_path)
        logger.debug(f"Using absolute path: {image_path_abs}")

        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Initialize the driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        wait = WebDriverWait(driver, 15)
        
        try:
            logger.function_call(f"Performing reverse image search for: {image_path}")
            
            # Navigate to Google Images
            driver.get("https://images.google.com")
            
            # Add random delay to appear more human-like
            delay = random.uniform(1, 3)
            logger.debug(f"Waiting {delay:.1f}s before interacting with page")
            time.sleep(delay)
            
            # Find and click the camera icon for reverse image search
            camera_button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "[aria-label='Search by image']"))
            )
            camera_button.click()
            logger.debug("Clicked camera button for reverse search")
            
            time.sleep(random.uniform(0.5, 1.5))
            
            # Find the file upload input
            file_input = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
            )
            
            # Upload the image file
            file_input.send_keys(image_path_abs)
            logger.debug("Uploaded image file")
            
            # Wait for the results page to load
            load_delay = random.uniform(4, 6)
            logger.debug(f"Waiting {load_delay:.1f}s for results to load")
            time.sleep(load_delay)
            
            logger.debug(f"Current URL: {driver.current_url}")
            
            results = []
            
            try:
                all_matches = driver.find_elements(By.CSS_SELECTOR, "div.srKDX.cvP2Ce > div")

                logger.debug(f"Found {len(all_matches)} matches on results page")
                
                for i, match in enumerate(all_matches):
                    try:
                        img = match.find_element(By.CSS_SELECTOR, "img[src]")
                        title = match.find_element(By.CSS_SELECTOR, "div.T3Fozb[aria-label]").get_attribute("aria-label")
                        url = match.find_elements(By.CSS_SELECTOR, "a[href]")[0].get_attribute("href")
                        
                        result = {
                            'id': i,
                            'title': title,
                            'url': url,
                            'img': img.get_attribute("src"),
                            'is_exact_match': (i == 0)  # True only for first result
                        }
                        
                        results.append(result)
                        logger.debug(f"Processed result {i+1}: {title[:50]}...")
                        
                        if len(results) >= 10:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Error processing match {i}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error finding search results: {e}")
            
            logger.function_call(f"Found {len(results)} reverse image search results")
            for i, result in enumerate(results[:3]):
                logger.debug(f"Result {i+1}: {result['title'][:50]}... -> {result['url'][:50]}...")
            
            # Save results to cache
            if results:
                save_cache(cache_path, results)
            else:
                logger.warning("No results found.")
                # with open("debug_page_source.html", "w", encoding="utf-8") as f:
                #     f.write(driver.page_source)

            for result in results:
                del result['img']
            
            return results[:10]
            
        finally:
            driver.quit()
            logger.debug("Closed Chrome driver")
            
    except Exception as e:
        logger.error(f"Error performing reverse image search: {str(e)}")
        return []
    

REVERSE_IMAGE_SEARCH_TOOL = {
    "name": "reverse_image_search",
    "description": "Performs reverse image search on an image file and shows the visual results. Can help with identifying the source of an image.",
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Name of the file to search for (e.g., 'X.jpg' or 'X.png')"
            }
        },
        "required": ["image_path"]
    }
}

GET_EXIF_TOOL = {
    "name": "get_exif_data",
    "description": "Extracts EXIF metadata from an image file, including camera settings, GPS coordinates, timestamps, and other technical information. Returns None if no EXIF data is found or if the file format doesn't support EXIF.",
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Name of the image file to extract EXIF data from (e.g., 'X.jpg' or 'X.png')"
            }
        },
        "required": ["image_path"]
    }
}

VIEW_IMAGE_FROM_REVERSE_IMAGE_SEARCH_TOOL = {
    "name": "view_image_from_reverse_image_search",
    "description": "Views an image from a reverse image search result.",
    "parameters": {
        "type": "object",
        "properties": {
            "case_image_id": {"type": "integer", "description": "ID of the image in the case (from the case metadata)"},
            "result_image_id": {"type": "integer", "description": "ID of the image to view (from the reverse image search results)"}
        },
        "required": ["case_image_id", "result_image_id"]
    }
}

GOOGLE_WEB_SEARCH_TOOL = {
    "name": "google_web_search",
    "description": "Search the web using Google",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"]
    }
}

VISIT_WEBSITE_TOOL = {
    "name": "visit_website",
    "description": "Visit a website and read it's title and full content (you cannot interact with the website, only extract information)",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to visit (can include or exclude protocol)"
            }
        },
        "required": ["url"]
    }
}

OVERPASS_TURBO_TOOL = {
    "name": "overpass_turbo_query",
    "description": "Execute Overpass QL queries to search OpenStreetMap data. Useful for finding geographic features, POIs, buildings, roads, etc. around specific locations or matching certain criteria.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Overpass QL query string. Examples: '(node[\"amenity\"=\"restaurant\"](around:1000,40.7128,-74.0060);); out;' or 'way[\"highway\"=\"primary\"][\"name\"~\"Broadway\"];out geom;'"
            }
        },
        "required": ["query"]
    }
}

GEOCODE_TOOL = {
    "name": "geocode",
    "description": "Geocode a location name or address using OpenStreetMap's Nominatim service. Converts location names to coordinates and detailed address information.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Location to search for (e.g., 'Niagara Falls', '123 Main St, Toronto', 'Eiffel Tower')"
            }
        },
        "required": ["query"]
    }
}

SV_QUERY_TOOL = {
    "name": "sv_query",
    "description": "Get Google Street View panorama image for specific coordinates (within 50 meter radius). Very useful for visual confirmation.",
    "parameters": {
        "type": "object",
        "properties": {
            "lat": {
                "type": "number",
                "description": "Latitude coordinate (e.g., 40.7128)"
            },
            "lng": {
                "type": "number", 
                "description": "Longitude coordinate (e.g., -74.0060)"
            }
        },
        "required": ["lat", "lng"]
    }
}

TOOLS_BASIC = [
    REVERSE_IMAGE_SEARCH_TOOL,
    GET_EXIF_TOOL,
    GOOGLE_WEB_SEARCH_TOOL,
    VISIT_WEBSITE_TOOL,
    GEOCODE_TOOL
]

TOOLS_BASIC_FULL = TOOLS_BASIC + [
    VIEW_IMAGE_FROM_REVERSE_IMAGE_SEARCH_TOOL,
    SV_QUERY_TOOL
]

TOOLS_ADVANCED = TOOLS_BASIC_FULL + [
    OVERPASS_TURBO_TOOL
]

if __name__ == "__main__":
    # your tool tests here

    set_dataset_path("dataset/name")
    reverse_image_search("1.jpg")