from fastapi import FastAPI, UploadFile, File, HTTPException
from predict import predict_single_server
from srcs import select_model, use_device, load_weights, load_categories
import argparse
import numpy as np
from PIL import Image
import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(description="Start FastAPI model server")
    parser.add_argument("--model", "-m", type=str, default="CNN",
                        help="Model type: CNN or RESNET")
    parser.add_argument("--loadweights", "-l", type=str, default="weights.pth",
                        help="Path to model weights")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--loadcategories", "-lc", type=str,
                        default="categories/categories.json")
    args = parser.parse_args()
    return args


def setup():
    args = parse_args()
    categories = load_categories(args.loadcategories)
    model = select_model(args.model, num_categories=len(categories))
    device = use_device(model)
    load_weights(model, args.loadweights, device)
    return args, model, device, categories


app = FastAPI()


@app.post("/leaffliction/predict")
def run_perdiction(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    img = Image.open(file.file).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    res = predict_single_server(arr, model, device, categories)
    print("The prediction is:", res["category"])
    return {"result": res}


if __name__ == "__main__":
    args, model, device, categories = setup()
    uvicorn.run(app, host=args.host, port=args.port)
