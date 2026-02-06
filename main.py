import cv2
import numpy as np
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional

# Force deploy 2026-02-06
app = FastAPI()

def order_points(pts):
    """
    Order points in clockwise order starting from top-left:
    top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    """
    Apply perspective transform to get bird's eye view of document
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calculate width of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Calculate height of new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Ensure minimum dimensions
    if maxWidth < 100 or maxHeight < 100:
        return None
    
    # Destination points for the transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Apply perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def detect_document(image):
    """
    Detect document edges using multiple methods for better accuracy
    """
    # Resize for processing (maintain aspect ratio)
    height, width = image.shape[:2]
    ratio = height / 500.0
    resized = cv2.resize(image, (int(width / ratio), 500))
    
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Method 1: Canny edge detection
    edged = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edged, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    document_contour = None
    
    # Find the largest contour with 4 corners
    for contour in contours:
        # Calculate perimeter and approximate polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # Check if it's a quadrilateral
        if len(approx) == 4:
            # Calculate area to filter out small contours
            area = cv2.contourArea(approx)
            image_area = resized.shape[0] * resized.shape[1]
            
            # Document should be at least 10% of image area
            if area > image_area * 0.1:
                document_contour = approx
                break
    
    # If no good contour found, try alternative method
    if document_contour is None:
        # Try adaptive thresholding approach
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        thresh = cv2.bitwise_not(thresh)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                image_area = resized.shape[0] * resized.shape[1]
                
                if area > image_area * 0.1:
                    document_contour = approx
                    break
    
    # Scale back to original size
    if document_contour is not None:
        return document_contour.reshape(4, 2) * ratio
    
    return None

def enhance_document(image):
    """
    Enhance scanned document for better readability
    """
    # Convert to grayscale for processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply adaptive thresholding for clean text
    # This works better than global thresholding for varying lighting
    enhanced = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 10
    )
    
    # Apply slight denoising
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Convert back to BGR for consistency
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced

def sharpen_image(image):
    """
    Apply sharpening to make text crisp
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

@app.post("/scan")
async def scan_image(
    file: UploadFile = File(...),
    enhance: bool = True,
    sharpen: bool = True
):
    """
    Scan a document image:
    - Detects document edges
    - Applies perspective transform
    - Enhances image quality
    
    Parameters:
    - file: Image file to scan
    - enhance: Apply adaptive thresholding for clean text (default: True)
    - sharpen: Apply sharpening filter (default: True)
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")
        
        # Validate image dimensions
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            raise HTTPException(status_code=400, detail="Image too small (minimum 100x100 pixels)")
        
        # Detect document edges
        document_points = detect_document(image)
        
        # Apply perspective transform if document detected
        if document_points is not None:
            processed = four_point_transform(image, document_points)
            
            if processed is None:
                # Fallback to original if transform failed
                processed = image
        else:
            # Use original image if no document detected
            processed = image
        
        # Apply enhancements
        if sharpen:
            processed = sharpen_image(processed)
        
        if enhance:
            processed = enhance_document(processed)
        else:
            # Basic contrast/brightness adjustment if not using adaptive threshold
            processed = cv2.convertScaleAbs(processed, alpha=1.2, beta=15)
        
        # Encode as high-quality JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        is_success, buffer = cv2.imencode(".jpg", processed, encode_params)
        
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode processed image")
        
        io_buf = io.BytesIO(buffer)
        
        return StreamingResponse(
            io_buf, 
            media_type="image/jpeg",
            headers={"X-Document-Detected": str(document_points is not None)}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.get("/")
def health_check():
    """Health check endpoint"""
    return {
        "status": "Scanner Service Running",
        "version": "2.0",
        "endpoints": {
            "/scan": "POST - Scan and process document images"
        }
    }

@app.get("/health")
def detailed_health():
    """Detailed health check with OpenCV info"""
    return {
        "status": "healthy",
        "opencv_version": cv2.__version__,
        "supported_formats": ["JPEG", "PNG", "BMP", "TIFF", "WebP"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)