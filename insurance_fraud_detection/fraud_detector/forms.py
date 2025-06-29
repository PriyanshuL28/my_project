from django import forms
from .models import FraudAnalysis

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = FraudAnalysis
        fields = ['uploaded_file']
        widgets = {
            'uploaded_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv'
            })
        }
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['uploaded_file'].label = 'Select CSV File'
        self.fields['uploaded_file'].help_text = 'Upload a CSV file containing insurance claims data (max 10MB)'
        
    def clean_uploaded_file(self):
        file = self.cleaned_data.get('uploaded_file')
        if file:
            if not file.name.endswith('.csv'):
                raise forms.ValidationError('Please upload a CSV file.')
            if file.size > 10485760:  # 10MB
                raise forms.ValidationError('File size must be under 10MB.')
        return file