import os
import torch

from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

# Import AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "dev-secret-key-change-me"
)

# Use absolute paths so Render/Linux can find the folders correctly
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["ALLOWED_EXTENSIONS"] = {
    "png",
    "jpg",
    "jpeg"
}

Bootstrap(app)

# Make sure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# UPLOAD FORM
# ============================================================

class UploadForm(FlaskForm):

    content = FileField("Content Image")
    style = FileField("Style Image")

    content_path = HiddenField()
    style_path = HiddenField()

    alpha = FloatField(
        "Alpha",
        default=1.0
    )

    submit = SubmitField("Transfer Style")


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("Device:", device)
print("========================================")


# ============================================================
# MODEL PATHS
# ============================================================

# Your actual folder structure is:
#
# Image-Transform/
# ├── app.py
# ├── vgg_normalised.pth
# ├── experiment/
# │   └── final_training/
# │       └── decoder_11.pth
# └── utils/
#     ├── models.py
#     └── utils.py

VGG_PATH = os.path.join(
    BASE_DIR,
    "vgg_normalised.pth"
)

MODEL_PATH = os.environ.get(
    "DECODER_PATH",
    os.path.join(
        BASE_DIR,
        "experiment",
        "final_training",
        "decoder_11.pth"
    )
)


print("VGG model path:")
print(VGG_PATH)

print("Decoder model path:")
print(MODEL_PATH)


# ============================================================
# CHECK MODEL FILES BEFORE LOADING
# ============================================================

if not os.path.isfile(VGG_PATH):
    raise FileNotFoundError(
        f"VGG model not found: {VGG_PATH}"
    )

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Decoder model not found: {MODEL_PATH}"
    )


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading VGG encoder...")

encoder = VGGEncoder(
    VGG_PATH
).to(device)

print("VGG encoder loaded successfully.")

print("Loading decoder...")

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

print("Decoder loaded successfully.")


# Evaluation mode
encoder.eval()
decoder.eval()

print("========================================")
print("Models loaded successfully!")
print("========================================")


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


# ============================================================
# STYLE TRANSFER
# ============================================================

def style_transfer(
    content_image,
    style_image,
    encoder,
    decoder,
    alpha,
    device
):

    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    # Convert images to tensors
    content_tensor = content_transform(
        content_image
    ).unsqueeze(0).to(device)

    style_tensor = style_transform(
        style_image
    ).unsqueeze(0).to(device)

    # Run model without calculating gradients
    with torch.no_grad():

        content_feats = encoder(
            content_tensor,
            is_test=True
        )

        style_feats = encoder(
            style_tensor,
            is_test=True
        )

        # Adaptive Instance Normalization
        stylized_feats = adaptive_instance_normalization(
            content_feats,
            style_feats
        )

        # Alpha controls style strength
        stylized_feats = (
            alpha * stylized_feats
            + (1 - alpha) * content_feats
        )

        # Decode stylized features
        stylized_image = decoder(
            stylized_feats
        )

    return stylized_image


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(image, path):

    image = image.cpu().clone()

    image = image.squeeze(0)

    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(image)

    image.save(path)


# ============================================================
# MAIN ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == "POST":

        if form.validate_on_submit():

            # ------------------------------------------------
            # CONTENT IMAGE
            # ------------------------------------------------

            if (
                form.content.data
                and form.content.data.filename
            ):

                if allowed_file(
                    form.content.data.filename
                ):

                    content_filename = secure_filename(
                        form.content.data.filename
                    )

                    content_path = os.path.join(
                        app.config["UPLOAD_FOLDER"],
                        content_filename
                    )

                    form.content.data.save(
                        content_path
                    )

                    form.content_path.data = (
                        content_filename
                    )

                else:

                    error = (
                        "Content image must be "
                        "png, jpg, or jpeg."
                    )

            else:

                content_filename = (
                    form.content_path.data
                )


            # ------------------------------------------------
            # STYLE IMAGE
            # ------------------------------------------------

            if (
                form.style.data
                and form.style.data.filename
            ):

                if allowed_file(
                    form.style.data.filename
                ):

                    style_filename = secure_filename(
                        form.style.data.filename
                    )

                    style_path = os.path.join(
                        app.config["UPLOAD_FOLDER"],
                        style_filename
                    )

                    form.style.data.save(
                        style_path
                    )

                    form.style_path.data = (
                        style_filename
                    )

                else:

                    error = (
                        "Style image must be "
                        "png, jpg, or jpeg."
                    )

            else:

                style_filename = (
                    form.style_path.data
                )


            # ------------------------------------------------
            # CHECK BOTH FILES
            # ------------------------------------------------

            if (
                not error
                and not (
                    content_filename
                    and style_filename
                )
            ):

                error = (
                    "Please upload both a "
                    "content image and a style image."
                )


            # ------------------------------------------------
            # STYLE TRANSFER
            # ------------------------------------------------

            if (
                not error
                and content_filename
                and style_filename
            ):

                content_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    content_filename
                )

                style_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    style_filename
                )

                try:

                    print("Opening content image...")

                    content_image = Image.open(
                        content_path
                    ).convert("RGB")

                    print("Opening style image...")

                    style_image = Image.open(
                        style_path
                    ).convert("RGB")

                    alpha = float(
                        form.alpha.data
                    )

                    # Keep alpha in a sensible range
                    alpha = max(
                        0.0,
                        min(1.0, alpha)
                    )

                    print(
                        f"Running style transfer "
                        f"with alpha={alpha}"
                    )

                    stylized_image = style_transfer(
                        content_image,
                        style_image,
                        encoder,
                        decoder,
                        alpha,
                        device
                    )

                    # ------------------------------------------------
                    # SAVE RESULT
                    # ------------------------------------------------

                    result_filename = (
                        "stylized_" + content_filename
                    )

                    result_path = os.path.join(
                        app.config["UPLOAD_FOLDER"],
                        result_filename
                    )

                    save_image(
                        stylized_image,
                        result_path
                    )

                    result_image = result_filename

                    print(
                        "Style transfer completed successfully."
                    )

                except Exception as e:

                    print(
                        "ERROR during style transfer:"
                    )

                    print(e)

                    error = str(e)

        else:

            error = (
                "There was a problem with your "
                "submission. Please try again."
            )

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# ============================================================
# SERVE UPLOADED IMAGES
# ============================================================

@app.route("/uploads/<filename>")
def send_image(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# SERVE EXAMPLE IMAGES
# ============================================================

@app.route("/examples/<path:filename>")
def send_example(filename):

    examples_folder = os.path.join(
        BASE_DIR,
        "examples"
    )

    return send_from_directory(
        examples_folder,
        filename
    )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
