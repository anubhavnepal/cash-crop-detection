import os
import io
import base64
import json
from django.db.models import Q
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.views.generic import TemplateView
from django.shortcuts import render, redirect
from django.utils.decorators import method_decorator
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.contrib import messages
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.http import require_GET
from django.contrib.auth import update_session_auth_hash
from django.db import IntegrityError
from django import forms
from django.contrib.auth.forms import AuthenticationForm
from .forms import PredictionForm, LoginForm, SignUpForm
from .models import Disease, Prediction
from PIL import Image
from model.predict import predict_disease
from plantapp.models import NearbyShop, SeasonalHarvesting, CropRotation, Crop


class UserRegistrationView(TemplateView):
    template_name = "register.html"

    def get(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect('/upload')
        return super(UserRegistrationView, self).get(request, *args, **kwargs)    

    def get_context_data(self, **kwargs):
        cxt = super().get_context_data(**kwargs)
        cxt['form'] = SignUpForm()
        return cxt

    def post(self, request):
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "User registered successfully. Please log in.")
            return redirect('login')
        else:
            messages.error(request, "Registration failed. Please check the form for errors.")
        return render(request, self.template_name, {"form": form})
class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({
            'class': 'input-field',
            'placeholder': 'Username'
        })
        self.fields['password'].widget.attrs.update({
            'class': 'input-field',
            'placeholder': 'Password'
        })
class LoginView(TemplateView):
    template_name = "login.html"

    def get(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect('upload')
        return render(request, self.template_name, {'form': LoginForm()})

    def post(self, request):
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            
            if user is not None:
                login(request, user)
                return redirect('upload')
        
        # Add non-field error if authentication fails
        form.add_error(None, "Invalid username or password")
        return render(request, self.template_name, {'form': form})

class PredictView(TemplateView):
    template_name = "dashboard/upload.html"

    def get_context_data(self, **kwargs):
        cxt = super().get_context_data(**kwargs)
        cxt['prediction_form'] = PredictionForm(None)

        if self.request.user.is_authenticated:
            load_all = self.request.GET.get('load_all', 'false').lower() == 'true'

            if load_all:
                past_records = Prediction.objects.filter(user=self.request.user).order_by('-date')
            else:
                past_records = Prediction.objects.filter(user=self.request.user).order_by('-date')[:4]

            cxt['past_records'] = past_records
            cxt['show_load_more'] = not load_all and Prediction.objects.filter(user=self.request.user).count() > 4

        return cxt

    def get_uploaded_image_and_predict(self, image):
        # Read the image data from the UploadedFile into memory
        image_data = image.read()
        
        # Convert to PIL Image
        im = Image.open(io.BytesIO(image_data))
        
        # Process image for preview
        data = io.BytesIO()
        im.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue()).decode('utf-8')
        
        # Reset the pointer to read the image data again for prediction
        image_stream = io.BytesIO(image_data)
        
        # Get prediction by passing the file-like object (BytesIO)
        prediction_result = predict_disease(image_stream)
        
        predicted_class = prediction_result["class"]
        
        return predicted_class, encoded_img_data

    def post(self, request):
        cxt = {}
        prediction_form = PredictionForm(request.POST, request.FILES)

        if prediction_form.is_valid():
            image = prediction_form.cleaned_data['image']

            prediction, template_image = self.get_uploaded_image_and_predict(image)

            if request.user.is_authenticated:
                Prediction.objects.create(
                    user=request.user,
                    image=image,
                    prediction=prediction,
                )

            def normalize_name(name):
                return name.replace("___", "_").replace("_", " ").strip().lower()

            normalized_prediction = normalize_name(prediction)

            diseases = Disease.objects.all()
            matched_disease = None

            for disease in diseases:
                if normalize_name(disease.name) == normalized_prediction:
                    matched_disease = disease
                    break

            if matched_disease:
                cxt['disease_name'] = matched_disease.name
                cxt['disease_description'] = matched_disease.description
                cxt['disease_area'] = matched_disease.area
                cxt['disease_amount'] = matched_disease.amount
                cxt['disease_protect'] = matched_disease.protect
                cxt['disease_severity'] = matched_disease.severity
                cxt['show_details_button'] = True
            else:
                cxt['disease_name'] = prediction
                cxt['disease_description'] = "Description not available for this disease."
                cxt['show_details_button'] = False

            cxt['prediction_form'] = PredictionForm(None)
            cxt['prediction'] = prediction
            cxt['image'] = template_image

            if request.user.is_authenticated:
                past_records = Prediction.objects.filter(user=request.user).order_by('-date')
                cxt['past_records'] = past_records
            
             # Check for AJAX request
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                # Render just the prediction results section
                html = render_to_string('prediction_results_partial.html', cxt)
                return JsonResponse({
                    'html': html,
                    'disease_available': bool(matched_disease)
                })
                
            return render(request, self.template_name, cxt)
    
    @method_decorator(login_required(login_url='/'))
    def dispatch(self, *args, **kwargs):
        return super(PredictView, self).dispatch(*args, **kwargs)

def delete_record(request, record_id):
    if request.user.is_authenticated:
        record = get_object_or_404(Prediction, id=record_id, user=request.user)
        record.delete()
        return HttpResponseRedirect(reverse('past_records'))
    else:
        return redirect('login')

def user_logout(request):
    logout(request)
    return redirect('landing_page')

class LandingPageView(TemplateView):
    template_name = "landing_page.html"


def past_records(request):
    if request.user.is_authenticated:
        load_all = request.GET.get('load_all', 'false').lower() == 'true'
        
        if load_all:
            records = Prediction.objects.filter(user=request.user).order_by('-date')
        else:
            records = Prediction.objects.filter(user=request.user).order_by('-date')[:4]
        
        show_load_more = not load_all and Prediction.objects.filter(user=request.user).count() > 4
        
        return render(request, 'dashboard/past_records.html', {'past_records': records, 'show_load_more': show_load_more})
    else:
        return redirect('login')

def get_image_mapping():
    return {
        'tea': 'images/tea-leaves.jpg',
        'potato': 'images/potato-leaves.jpg',
        'sugarcane': 'images/sugarcane-leaves.jpeg',
        'coffee': 'images/coffee-leaves.jpg'
    }

@login_required
def crops_view(request):
    image_mapping = get_image_mapping()
    crops = [
        {'name': 'Tea', 'slug': 'tea', 'image_src': image_mapping['tea']},
        {'name': 'Potato', 'slug': 'potato', 'image_src': image_mapping['potato']},
        {'name': 'Sugarcane', 'slug': 'sugarcane', 'image_src': image_mapping['sugarcane']},
        {'name': 'Coffee', 'slug': 'coffee', 'image_src': image_mapping['coffee']},
    ]
    return render(request, 'dashboard/crops.html', {'crops': crops})

@login_required
def crop_detail(request, crop_name):
    """
    Shows a detail page for the given crop (e.g. 'tea', 'potato', etc.).
    We assume Disease.name has patterns like 'Tea Healthy', 'Tea Bird Eye Spot', etc.
    """

    healthy_disease = Disease.objects.filter(
        Q(name__icontains=crop_name) & Q(name__icontains='healthy')
    ).first()

    #Get the disease entries (exclude the healthy one).
    disease_list = Disease.objects.filter(Q(name__icontains=crop_name) & ~Q(name__icontains='healthy'))

    # Add dynamic image mapping based on crop_name
    image_mapping = get_image_mapping()
    image_src = image_mapping.get(crop_name.lower(), 'images/cash_crops.jpg')
    
    context = {
        'crop_name': crop_name.capitalize(),
        'healthy_disease': healthy_disease,   # might be None if not found
        'diseases': disease_list,
        'image_src': image_src,  # Pass the dynamic image path
    }
    return render(request, 'dashboard/crop_detail.html', context)

@login_required
def shop_view(request):
    return render(request, 'dashboard/nearby_shop.html')

def shop_data(request):
    shops = NearbyShop.objects.all()
    shop_list = []
    for shop in shops:
        shop_list.append({
            'name': shop.name,
            'address': shop.address,
            'phone': shop.phone,
            'latitude': shop.latitude,
            'longitude': shop.longitude,
            'open_time': shop.open_time.strftime('%H:%M'),
            'close_time': shop.close_time.strftime('%H:%M'),
        })
    return JsonResponse({'shops': shop_list})

@login_required
def seasonal_harvest_view(request):
    return render(request, 'dashboard/seasonal_harvest.html')

def seasonal_harvesting_api(request):
    crops = SeasonalHarvesting.objects.all()
    crops_data = []
    for crop in crops:
        crops_data.append({
            "title": crop.title,
            "icon": crop.icon or "",
            "season": crop.season,       # Already a list of dicts
            "techniques": crop.techniques  # Already a list of strings
        })
    return JsonResponse(crops_data, safe=False)

@login_required
def farming_cost_calculator(request):
    return render(request, 'dashboard/cost_calculator.html')

@login_required
def get_crop_rotation_data(request):
    crops = CropRotation.objects.all()
    crop_list=[
        {
            "name": crop.name,
            "suitable_months": crop.suitable_months,
            "season_text": crop.season_text
        }
        for crop in crops
    ]
    return JsonResponse(crop_list, safe=False)

def get_crop_growth_duration(request):
    crop_name = request.GET.get('crop')
    try:
        crop = Crop.objects.get(name=crop_name)
        return JsonResponse({
            'growth_duration': crop.growth_duration,
            'water_per_day': crop.water_per_day 
        })
    except Crop.DoesNotExist:
        return JsonResponse({'error': 'Crop not found'}, status=404)

@login_required
def profile_view(request):
    if request.method == 'POST':
        user = request.user
        new_username = request.POST.get('username')
        current_password = request.POST.get('current_password')
        new_password1 = request.POST.get('new_password1')
        new_password2 = request.POST.get('new_password2')

        # Validate current password
        if not user.check_password(current_password):
            messages.error(request, 'Current password is incorrect')
            return redirect('user_profile')

        # Check for username uniqueness 
        if new_username and new_username != user.username:
            if User.objects.filter(username=new_username).exists():
                messages.error(request, 'Username already exists. Please choose a different one.')
                return redirect('user_profile')

        # if validation true, Update username 
        if new_username and new_username != user.username:
            user.username = new_username
            messages.success(request, 'Username updated successfully')

        # Password change logic
        if new_password1 and new_password2:
            if new_password1 == new_password2:
                user.set_password(new_password1)
                update_session_auth_hash(request, user)
                messages.success(request, 'Password updated successfully')
            else:
                messages.error(request, 'New passwords do not match')
                return redirect('user_profile')

        try:
            user.save()
        except IntegrityError:
            messages.error(request, 'Username already exists. Please choose a different one.')
            return redirect('user_profile')

        return redirect('user_profile')

    return render(request, 'dashboard/user_profile.html', {'user': request.user})
