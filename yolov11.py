from ultralytics import YOLO

# # Load a YOLO11n PyTorch model
# model = YOLO("yolo11n-seg.pt")
#
# # Export the model to NCNN format
# model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'

# Load the exported NCNN model
ncnn_model = YOLO("yolo11n-seg_ncnn_model")

# Run inference
results = ncnn_model("test.jpg")

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.save(filename="resulttest.jpg")  # save to disk