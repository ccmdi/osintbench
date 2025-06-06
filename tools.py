def get_exif_data(image_path: str) -> dict:
    import piexif
    import os

    image_path = os.path.join("dataset", "basic", "images", image_path)

    try:
        exif_dict = piexif.load(image_path, True)
        
        if 'thumbnail' in exif_dict:
            del exif_dict['thumbnail']

        return exif_dict
    except Exception as e:
        if "No EXIF data found" in str(e) or "Given file is neither JPEG nor TIFF" in str(e):
            print(f"No EXIF data found for {image_path}")
            return None
        else:
            print(f"Error extracting EXIF data: {str(e)}")
            return None


def visual_reverse_image_search(image_path: str, use_cache: bool = False) -> dict:
    """
    Performs a reverse image search and returns the first result's image for comparison.
    
    Args:
        image_path (str): Path to the image file (e.g., "dataset/basic/images/16.jpg")
        use_cache (bool): Whether to use cached results if available
        
    Returns:
        dict: Contains the first search result's image for visual comparison
    """
    import requests
    import base64
    from PIL import Image
    import io
    import os
    from playwright.sync_api import sync_playwright
    
    image_path = os.path.join("dataset", "basic", "images", image_path)
    
    # Get the basic search results using your existing function
    search_results = reverse_image_search_results(image_path, use_cache)
    
    if not search_results:
        return {
            "success": False,
            "message": "No reverse image search results found",
            "original_image": image_path
        }
    
    print(f"🎯 Found {len(search_results)} search results, looking for images...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            # Try each search result to find an image
            for i, result in enumerate(search_results[:5]):
                print(f"🔍 Checking result {i+1}: {result['title'][:50]}...")
                print(f"📍 URL: {result['url']}")
                
                # First try if it's a direct image URL
                if any(result['url'].lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']):
                    try:
                        response = requests.get(
                            result['url'], 
                            timeout=15,
                            headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            }
                        )
                        
                        if response.status_code == 200 and response.headers.get('content-type', '').startswith('image/'):
                            image_b64 = base64.b64encode(response.content).decode()
                            image = Image.open(io.BytesIO(response.content))
                            
                            print(f"✅ Found direct image: {image.size} pixels, {image.format}")
                            
                            return {
                                "success": True,
                                "original_image": image_path,
                                "found_image": image_b64,
                                "image_info": {
                                    "size": image.size,
                                    "format": image.format,
                                    "source_url": result['url'],
                                    "source_title": result['title']
                                }
                            }
                    except Exception as e:
                        print(f"❌ Direct image failed: {str(e)}")
                        continue
                
                # Try to extract image from the webpage
                try:
                    print("🌐 Visiting webpage to extract images...")
                    page.goto(result['url'], timeout=20000, wait_until='domcontentloaded')
                    page.wait_for_timeout(2000)  # Wait for dynamic content
                    
                    # Get all images on the page
                    images = page.query_selector_all('img')
                    
                    for img in images:
                        try:
                            img_src = img.get_attribute('src')
                            if not img_src:
                                continue
                            
                            # Handle relative URLs
                            if img_src.startswith('//'):
                                img_src = 'https:' + img_src
                            elif img_src.startswith('/'):
                                from urllib.parse import urljoin
                                img_src = urljoin(result['url'], img_src)
                            elif not img_src.startswith('http'):
                                continue
                            
                            # Try to download the image
                            response = requests.get(
                                img_src,
                                timeout=10,
                                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                            )
                            
                            if (response.status_code == 200 and 
                                response.headers.get('content-type', '').startswith('image/') and
                                len(response.content) > 10000):  # Filter out tiny images
                                
                                image = Image.open(io.BytesIO(response.content))
                                if image.size[0] > 200 and image.size[1] > 200:  # Reasonable size
                                    
                                    image_b64 = base64.b64encode(response.content).decode()
                                    
                                    print(f"✅ Found image from webpage: {image.size} pixels, {image.format}")
                                    
                                    return {
                                        "success": True,
                                        "original_image": image_path,
                                        "found_image": image_b64,
                                        "image_info": {
                                            "size": image.size,
                                            "format": image.format,
                                            "source_url": img_src,
                                            "source_page": result['url'],
                                            "source_title": result['title']
                                        }
                                    }
                        
                        except Exception as e:
                            continue  # Try next image
                            
                except Exception as e:
                    print(f"❌ Error accessing page {result['url']}: {str(e)}")
                    continue
            
            # No images found
            return {
                "success": False,
                "message": "No accessible images found in search results",
                "original_image": image_path,
                "search_results_checked": [r['url'] for r in search_results[:5]]
            }
            
        finally:
            browser.close()


def reverse_image_search_results(image_path: str, use_cache: bool = False) -> list:
    #TODO: this whole function is WRONG and BAD
    # the serach results arent correct to what i see when i do it so
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
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    import os
    import time
    import random
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin, urlparse
    import json
    
    def get_cache_path(image_path: str) -> str:
        """Generate cache file path based on image path"""
        # Parse the image path: dataset/basic/images/16.jpg -> dataset/basic/reverse-image-cache/16.txt
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
                    print(f"✓ Loaded {len(cached_data)} results from cache: {cache_path}")
                    return cached_data
        except Exception as e:
            print(f"Warning: Could not load cache from {cache_path}: {e}")
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
            
            print(f"✓ Saved {len(results)} results to cache: {cache_path}")
            
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_path}: {e}")
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return []
    
    # Get cache path and check for cached results
    cache_path = get_cache_path(image_path)
    
    if use_cache:
        cached_results = load_cache(cache_path)
        if cached_results:
            return cached_results
    
    try:
        # Get absolute path for the image
        image_path_abs = os.path.abspath(image_path)
        
        # Setup Chrome options for headless browsing
        chrome_options = Options()
        chrome_options.add_argument("--headless")
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
            print(f"🔍 Performing reverse image search for: {image_path}")
            
            # Navigate to Google Images
            driver.get("https://images.google.com")
            
            # Add random delay to appear more human-like
            time.sleep(random.uniform(1, 3))
            
            # Find and click the camera icon for reverse image search
            camera_button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "[aria-label='Search by image']"))
            )
            camera_button.click()
            
            time.sleep(random.uniform(0.5, 1.5))
            
            # Find the file upload input
            file_input = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
            )
            
            # Upload the image file
            file_input.send_keys(image_path_abs)
            
            # Wait for the results page to load
            time.sleep(random.uniform(4, 6))
            
            print("Current URL:", driver.current_url)
            
            results = []
            
            # Find external links in the search results
            # Google shows results in various containers, we'll look for links that lead to external sites
            try:
                # Look for all clickable links that are not Google internal links
                all_links = driver.find_elements(By.CSS_SELECTOR, "a[href]")
                
                for link in all_links:
                    try:
                        url = link.get_attribute("href")
                        title = link.text.strip()
                        
                        # Filter for external links only (exclude Google internal links)
                        if (url and title and 
                            url.startswith("http") and 
                            "google.com" not in url and 
                            "googleusercontent.com" not in url and
                            "gstatic.com" not in url and
                            len(title) > 3 and
                            len(title) < 200):  # Reasonable title length
                            
                            # Skip if we already have this URL
                            if any(result['url'] == url for result in results):
                                continue
                                
                            result = {
                                'title': title,
                                'url': url,
                                'description': '',
                                'source': 'Google Images Reverse Search'
                            }
                            
                            results.append(result)
                            
                            if len(results) >= 10:
                                break
                                
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"Error finding search results: {e}")
            
            # Debug: Print what we found
            print(f"Found {len(results)} results")
            for i, result in enumerate(results[:3]):
                print(f"Result {i+1}: {result['title'][:50]}... -> {result['url'][:50]}...")
            
            # Save results to cache
            if results:
                save_cache(cache_path, results)
            else:
                print("No results found. Saving page source for debugging...")
                with open("debug_page_source.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                print("Page source saved to debug_page_source.html")
            
            return results[:10]  # Ensure we return max 10 results
            
        finally:
            # Always close the driver
            driver.quit()
            
    except Exception as e:
        print(f"Error performing reverse image search: {str(e)}")
        return []
    

VISUAL_REVERSE_SEARCH_TOOL = {
    "name": "visual_reverse_image_search",
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


TOOLS = []