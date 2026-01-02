import cv2
import numpy as np
import time
from dataclasses import dataclass

@dataclass
class FoundObject:
    polygon : tuple = None  # ((x1,y1), (x2,y2), (x3,y3), (x4,y4), ...) vertices of bounding polygon
    norm_polygon : tuple = None  # normalized vertices
    centroid : tuple = None  # (x, y)
    norm_centroid : tuple = None  # normalized (x, y)
    area : float = None  # polygon area in pixels
    timestamp : float = None  # time.time() when object was detected
    source_finder : 'ObjectFinder' = None
    source_image = None  # The original image this object was detected in
    
    def compare_to(self, other):
        """Compare similarity to another FoundObject (0.0-1.0), or None if incompatible types."""
        if type(self) != type(other):
            return None  # Can't compare different object types (e.g., ArucubeObject vs YoloObject)
        return self.source_finder.score_similarity(self, other)
    
    def is_identical_to(self, other):
        """Check if this object is identical to another based on identifying attributes.
        
        Returns True if the objects are indistinguishable, False if they differ,
        or None if they are incompatible types.
        """
        if type(self) != type(other):
            return None
        return self.source_finder.are_identical(self, other)

class ObjectFinder:
    object_class = FoundObject
    fields_to_ignore = {'polygon', 'norm_polygon', 'centroid', 'norm_centroid', 
                        'area', 'timestamp', 'source_finder', 'source_image'}
    identity_threshold = 1.0  # Similarity threshold to consider objects as the same (0.0-1.0)
    
    def _find(self, rgb_image, drawing_image=None):
        return []  # To be implemented by subclasses
    
    def find(self, rgb_image, drawing_image=None):
        detections = self._find(rgb_image, drawing_image)
        if detections is None or rgb_image is None:
            return []
        
        height, width = rgb_image.shape[:2]
        results = []
        
        for detection in detections:
            if isinstance(detection, tuple) and len(detection) > 0 and isinstance(detection[0], tuple):
                polygon = detection
                obj = self.object_class(polygon=polygon)
            else:
                obj = detection
                polygon = obj.polygon
            
            # Compute centroid as average of all vertices
            centroid_x = sum(pt[0] for pt in polygon) / len(polygon)
            centroid_y = sum(pt[1] for pt in polygon) / len(polygon)
            centroid = (centroid_x, centroid_y)
            
            # Normalize polygon vertices
            norm_polygon = tuple(
                ((x - width / 2) / (width / 2), (y - height / 2) / (height / 2))
                for x, y in polygon
            )
            
            # Normalize centroid
            norm_centroid_x = (centroid[0] - width / 2) / (width / 2)
            norm_centroid_y = (centroid[1] - height / 2) / (height / 2)
            norm_centroid = (norm_centroid_x, norm_centroid_y)
            
            # Compute area
            area = cv2.contourArea(np.array(polygon, dtype=np.float32))
            
            # Get current timestamp
            timestamp = time.time()
            
            obj.polygon = polygon
            obj.norm_polygon = norm_polygon
            obj.centroid = centroid
            obj.norm_centroid = norm_centroid
            obj.area = area
            obj.timestamp = timestamp
            obj.source_finder = self
            obj.source_image = rgb_image
            
            results.append(obj)
        
        return results

    def score_similarity(self, obj1, obj2):
        """Score similarity 0.0-1.0 based on identifying fields (override for custom logic)."""
        from dataclasses import fields
        
        all_fields = {f.name for f in fields(obj1)}
        fields_to_ignore = type(self).fields_to_ignore
        identifying_fields = all_fields - fields_to_ignore
        
        if not identifying_fields:
            return 1.0 if obj1 == obj2 else 0.0
        
        matches = sum(getattr(obj1, f) == getattr(obj2, f) 
                     for f in identifying_fields)
        return matches / len(identifying_fields)
    
    def are_identical(self, obj1, obj2):
        """Check if two objects are indistinguishable based on their identifying attributes.
        
        Returns True if the objects cannot be differentiated given the available information
        (i.e., similarity >= identity_threshold). This does NOT guarantee they are the same
        real-world object, only that they are identical within the resolution of our detection.
        """
        similarity = self.score_similarity(obj1, obj2)
        return similarity >= type(self).identity_threshold

@dataclass
class ArucoObject(FoundObject):
    id: int | None = None  # ArUco marker ID
    dictionary: int | None = None  # ArUco dictionary enum value

class ArucoFinder(ObjectFinder):
    object_class = ArucoObject
    
    def __init__(self, dictionary=None, ids=None):
        """
        Args:
            dictionary: ArUco dictionary enum (e.g., cv2.aruco.DICT_4X4_50)
            ids: List of marker IDs to detect, or None to detect all
        """
        if dictionary is None:
            dictionary = cv2.aruco.DICT_4X4_50
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.ids_to_detect = ids
        self.dictionary_enum = dictionary
    
    def _find(self, rgb_image, drawing_image=None):
        corners, ids, _ = self.detector.detectMarkers(rgb_image)
        
        if ids is None:
            return []
        
        # Filter corners and ids based on ids_to_detect
        filtered_corners = []
        filtered_ids = []
        for i, marker_id in enumerate(ids.flatten()):
            if self.ids_to_detect is None or marker_id in self.ids_to_detect:
                filtered_corners.append(corners[i])
                filtered_ids.append(marker_id)
        
        # Early return if no markers match filter
        if not filtered_corners:
            return []
        
        # Convert filtered_ids back to numpy array for drawing
        filtered_ids_array = np.array(filtered_ids).reshape(-1, 1)
        
        # Draw only filtered markers on the drawing image if provided
        if drawing_image is not None:
            cv2.aruco.drawDetectedMarkers(drawing_image, filtered_corners, filtered_ids_array)
        
        # Create ArucoObjects from filtered corners and ids
        detections = []
        for i, marker_id in enumerate(filtered_ids):
            corner = filtered_corners[i][0]
            polygon = tuple((float(pt[0]), float(pt[1])) for pt in corner)
            
            obj = ArucoObject(polygon=polygon, id=int(marker_id), dictionary=self.dictionary_enum)
            detections.append(obj)
        
        return detections

class MultiFinder(ObjectFinder):
    def __init__(self, finders):
        self.finders = finders
    
    def find(self, rgb_image, drawing_image=None):
        """Override find to call each finder's full find() method to preserve source_finder."""
        all_detections = []
        for finder in self.finders:
            detections = finder.find(rgb_image, drawing_image)
            all_detections.extend(detections)
        return all_detections

# Common Helpers
def get_largest(img_objects):
    if not img_objects:
        return None
    return max(img_objects, key=lambda obj: obj.area)

if __name__ == "__main__":
    from arucube_finder import ArucubeFinder
    from yolo_finder import YoloFinder

    # Create object finder
    finder = MultiFinder([
        ArucubeFinder(ArucoFinder(dictionary=cv2.aruco.DICT_4X4_50, ids=[0, 1, 2])),
        YoloFinder(weights="tennis.pt", conf=0.40),
    ])

    # Test with a sample image
    test_image = cv2.imread("images/img1.jpeg")
    
    if test_image is None:
        print("Error: Could not load test image")
        exit(1)
    
    # Create drawing frame
    drawing_frame = test_image.copy()
    
    # Find all objects
    found_objects = finder.find(test_image, drawing_frame)
    
    print(f"Found {len(found_objects)} objects:")
    for i, obj in enumerate(found_objects):
        obj_type = type(obj).__name__
        print(f"  [{i}] {obj_type}: centroid={obj.centroid}, area={obj.area:.1f}")
        
        # Print type-specific info
        if hasattr(obj, 'cube_id'):
            print(f"      Cube ID: {obj.cube_id}, faces visible: {obj.marker_count}")
        elif hasattr(obj, 'id'):
            print(f"      ArUco ID: {obj.id}")
        elif hasattr(obj, 'class_name'):
            print(f"      Class: {obj.class_name}, confidence: {obj.confidence:.2f}")
    
    # Compare first two objects if available
    if len(found_objects) >= 2:
        obj1, obj2 = found_objects[0], found_objects[1]
        similarity = obj1.compare_to(obj2)
        identical = obj1.is_identical_to(obj2)
        
        print(f"\nComparing objects [0] and [1]:")
        print(f"  Object [0]: {type(obj1).__name__}")
        print(f"  Object [1]: {type(obj2).__name__}")
        print(f"  Similarity score: {similarity}")
        print(f"  Are identical: {identical}")
        
        if similarity is None:
            print("  -> Cannot compare (different object types)")
        elif identical:
            print("  -> Identical (indistinguishable)")
        elif similarity > 0.0:
            print(f"  -> Partial match ({similarity*100:.0f}% similar)")
        else:
            print("  -> No match (different objects)")
    else:
        print(f"\nNeed at least 2 objects to compare (found {len(found_objects)})")
    
    # Display result
    cv2.imshow("Detections", drawing_frame)
    print("\nPress any key to exit...")
    
    try:
        while True:
            key = cv2.waitKey(100)
            if key != -1:  # Any key pressed
                break
            # Check if window was closed
            if cv2.getWindowProperty("Detections", cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()