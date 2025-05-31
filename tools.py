# TODO: :)

def get_exif_data(image_path: str) -> dict:
    import piexif

    try:
        exif_dict = piexif.load(image_path, True)
        return exif_dict
    except Exception as e:
        print(f"Error extracting EXIF data: {str(e)}")
        return {}