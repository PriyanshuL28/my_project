from django.db import models
from django.contrib.auth.models import User
import json

class UploadedFile(models.Model):
    """Track all uploaded files"""
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_name = models.CharField(max_length=255)
    file_size = models.IntegerField()
    
    def __str__(self):
        return f"{self.file_name} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"

class FraudAnalysis(models.Model):
    uploaded_file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    # Summary statistics
    total_claims = models.IntegerField(default=0)
    low_risk_count = models.IntegerField(default=0)
    medium_risk_count = models.IntegerField(default=0)
    high_risk_count = models.IntegerField(default=0)
    critical_risk_count = models.IntegerField(default=0)
    
    # File paths
    output_csv_path = models.CharField(max_length=500, blank=True)
    high_risk_csv_path = models.CharField(max_length=500, blank=True)
    
    # Visualization paths
    visualizations = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
        
    def __str__(self):
        return f"Analysis {self.id} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"

class Claim(models.Model):
    RISK_LEVELS = [
        ('Low', 'Low Risk'),
        ('Medium', 'Medium Risk'),
        ('High', 'High Risk'),
        ('Critical', 'Critical Risk'),
    ]
    
    analysis = models.ForeignKey(FraudAnalysis, on_delete=models.CASCADE, related_name='claims')
    claim_number = models.CharField(max_length=100)
    claimant_name = models.CharField(max_length=200)
    date_of_loss = models.DateField(null=True, blank=True)
    injury_type = models.CharField(max_length=200, blank=True)
    body_part = models.CharField(max_length=200, blank=True)
    
    # Fraud indicators
    fraud_score = models.FloatField(default=0)
    risk_level = models.CharField(max_length=20, choices=RISK_LEVELS, default='Low')
    red_flags = models.JSONField(default=list)
    
    # Additional fields
    days_to_report = models.IntegerField(null=True, blank=True)
    claim_amount = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    
    class Meta:
        ordering = ['-fraud_score']
        
    def __str__(self):
        return f"Claim {self.claim_number} - {self.risk_level} Risk"