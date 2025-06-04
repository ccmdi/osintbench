def get_exif_data(image_path: str) -> dict:
    import piexif

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


def visit_webpage(url: str) -> str:
    #TODO
    pass

def reverse_image_search(image_path: str, use_cache: bool = True) -> list:
    """
    Performs a reverse image search using Google Images and returns the first 10 results.
    
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
            time.sleep(random.uniform(3, 5))
            
            # Wait for search results to appear - reverse image search has different structure
            time.sleep(3)  # Give more time for results to load
            
            # Debug: Print page source to see what we're working with
            print("Current URL:", driver.current_url)
            
            results = []
            
            # First, try to find "Pages that include matching images" section
            try:
                # Look for the section with web results that contain the image
                web_results_section = driver.find_elements(By.XPATH, "//h3[contains(text(), 'Pages that include matching images')]/following-sibling::div//a[@href]")
                
                for link in web_results_section[:10]:
                    try:
                        url = link.get_attribute("href")
                        title = link.text.strip()
                        
                        if url and title and url.startswith("http") and "google.com" not in url:
                            result = {
                                'title': title,
                                'url': url,
                                'description': '',
                                'source': 'Google Images - Pages with image'
                            }
                            
                            results.append(result)
                    except:
                        continue
                        
            except Exception as e:
                print(f"Error finding web results section: {e}")
            
            # If no results yet, try finding visually similar images section
            if not results:
                try:
                    # Look for "Visually similar images" or related images
                    similar_images = driver.find_elements(By.CSS_SELECTOR, "div[data-lpage] a[href]")
                    
                    for link in similar_images[:10]:
                        try:
                            url = link.get_attribute("href")
                            # Get title from image alt text or nearby text
                            title = link.get_attribute("title") or link.get_attribute("aria-label") or "Similar image"
                            
                            if url and url.startswith("http") and "google.com" not in url:
                                result = {
                                    'title': title,
                                    'url': url,
                                    'description': '',
                                    'source': 'Google Images - Similar image'
                                }
                                
                                results.append(result)
                        except:
                            continue
                            
                except Exception as e:
                    print(f"Error finding similar images: {e}")
            
            # Fallback: Try to find ANY external links on the page
            if not results:
                try:
                    print("Trying fallback method - finding all external links")
                    all_links = driver.find_elements(By.CSS_SELECTOR, "a[href]")
                    
                    for link in all_links:
                        try:
                            url = link.get_attribute("href")
                            title = link.text.strip()
                            
                            # Filter for external links only
                            if (url and title and 
                                url.startswith("http") and 
                                "google.com" not in url and 
                                "googleusercontent.com" not in url and
                                len(title) > 3):
                                
                                result = {
                                    'title': title,
                                    'url': url,
                                    'description': '',
                                    'source': 'Google Images - External link'
                                }
                                
                                results.append(result)
                                
                                if len(results) >= 10:
                                    break
                        except:
                            continue
                            
                except Exception as e:
                    print(f"Error in fallback method: {e}")
            
            # Debug: Print what we found
            print(f"Found {len(results)} results")
            for i, result in enumerate(results[:3]):
                print(f"Result {i+1}: {result['title'][:50]}... -> {result['url'][:50]}...")
            
            # Save results to cache
            if results:
                save_cache(cache_path, results)
            
            # If still no results, save page source for debugging
            if not results:
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
    
#TODO
# TOOLS = {
#     []
# }