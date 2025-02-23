document.addEventListener('DOMContentLoaded', function() {
  // Image preview logic
  document.getElementById('id_image').addEventListener('change', function(e) {
      const preview = document.getElementById('preview');
      if (this.files && this.files[0]) {
          preview.src = URL.createObjectURL(this.files[0]);
          document.getElementById('image-preview').style.display = 'block';
      }
  });
});