import os
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


app = Flask(__name__)
# Prefer an env var in real deployments; falls back to a dev-only default.
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Jo uploaded images hai vo store here
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}  # Jo data upload ho rha h it should be image only

Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Basically exist krta hai to koi nahi okey error na de


# class for upload form ki upload vagera manage
class UploadForm(FlaskForm):
    content = FileField('Content Image')   # Content and style images ke field ka naam and their path
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIX: raw string (r'...') so backslashes aren't parsed as escape sequences.
# Without the "r" prefix, "\f" in "\final_training" was silently turned into
# a form-feed character, corrupting the path (that's what caused the
# "OSError: [Errno 22] Invalid argument" you hit).
MODEL_PATH = os.environ.get(
    'DECODER_PATH',
    r'C:\Final-Projects\Image-Transform\experiment\final_training\decoder_11.pth'
)

encoder = VGGEncoder('vgg_normalised.pth').to(device)
decoder = Decoder().to(device)
# FIX: map_location=device so this still loads on a machine/session without a GPU.
decoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))

encoder.eval()
decoder.eval()


# This is checking jo file humne daali hai allowed hai ya nahi
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']  # Basically right part after . should be jpg,png or something


def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


# ------------------ Main routes of app ------------------ #
@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == 'POST' and form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
            else:
                error = 'Content image must be a png, jpg, or jpeg file'
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
            else:
                error = 'Style image must be a png, jpg, or jpeg file'
        else:
            style_filename = form.style_path.data

        # FIX: this case was previously silent — validate_on_submit() can be True
        # even with no files attached, so nothing told the user what was missing.
        if not error and not (content_filename and style_filename):
            error = 'Please upload both a content image and a style image'

        if not error and content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')

                alpha = float(form.alpha.data)
                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)

                result_filename = 'stylized_' + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)

                result_image = result_filename
            except Exception as e:
                error = str(e)
    elif request.method == 'POST':
        error = 'There was a problem with your submission. Please try again.'

    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                            style_image=style_filename, error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)