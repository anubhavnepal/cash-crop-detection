from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from django.utils.html import format_html

from .models import Disease, Prediction, NearbyShop, SeasonalHarvesting, CropRotation, Crop

# Disease model registration
admin.site.register(Disease)
admin.site.register(NearbyShop)
admin.site.register(SeasonalHarvesting)
admin.site.register(CropRotation)
admin.site.register(Crop)

# Prediction model registration
@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('user', 'prediction', 'image_preview', 'date')
    search_fields = ('user__username', 'prediction')
    list_filter = ('user', 'date')

    def image_preview(self, obj):
        if obj.image:
            return format_html('<img src="{}" width="100" height="100" />', obj.image.url)
        return "No Image"
    image_preview.short_description = "Image Preview"


# Add predictions inline to the User detail page
class PredictionInline(admin.TabularInline):
    model = Prediction
    extra = 0  
    fields = ('prediction', 'image', 'date')  
    readonly_fields = ('prediction', 'image', 'date') 


# Customize UserAdmin to include predictions inline
class CustomUserAdmin(UserAdmin):
    inlines = [PredictionInline]


# Unregistering default User admin and register the customized one
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)
