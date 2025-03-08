from django.urls import path, include
from django.contrib.auth.views import LogoutView
from django.conf import settings
from django.conf.urls.static import static


from .views import PredictView, LoginView, UserRegistrationView, user_logout, LandingPageView, past_records, profile_view, crops_view, shop_view, shop_data, seasonal_harvest_view, farming_cost_calculator, get_crop_growth_duration
from plantapp import views

urlpatterns = [
    # Auth views
    path('', LandingPageView.as_view(), name="landing_page"),
    path('login', LoginView.as_view(), name="login"),
    path('register', UserRegistrationView.as_view(), name="register"),
    path('logout', user_logout, name="logout"),

    # Custom views
    path('index/', LandingPageView.as_view(), name="index"), 
    
    # Sidebar views
    path('crops/', crops_view, name='crops'),
    path('nearby-shop/', shop_view, name='nearby-shop'),
    path('seasonal-harvest/', seasonal_harvest_view, name='seasonal-harvest'),
    path('optimization_tools/', farming_cost_calculator, name='cost-calculator'),
    path('upload/', PredictView.as_view(), name="upload"),
    path('past-records/', past_records, name='past_records'),
    path('user-profile/', profile_view, name='user_profile'),

    path('api/shops/', shop_data, name='shop_data'),
    path('api/seasonal-harvesting/', views.seasonal_harvesting_api, name='seasonal_harvesting_api'),
    path('api/crop-rotation/', views.get_crop_rotation_data, name='crop_rotation_api'),
    path('api/get_crop_growth_duration/', get_crop_growth_duration, name='get_crop_growth_duration'),

    path('crops/<str:crop_name>/', views.crop_detail, name='crop_detail'),
    
    # Delete Records
    path('delete-record/<int:record_id>/', views.delete_record, name='delete_record'),
]

# Serving media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)