from django.contrib import admin
from .models import FraudAnalysis, Claim

@admin.register(FraudAnalysis)
class FraudAnalysisAdmin(admin.ModelAdmin):
    list_display = ['id', 'uploaded_at', 'processed_at', 'total_claims', 
                   'high_risk_count', 'critical_risk_count', 'user']
    list_filter = ['uploaded_at', 'processed_at']
    search_fields = ['user__username']
    readonly_fields = ['uploaded_at', 'processed_at', 'visualizations']
    
    fieldsets = (
        ('Upload Information', {
            'fields': ('uploaded_file', 'uploaded_at', 'user')
        }),
        ('Processing Results', {
            'fields': ('processed_at', 'total_claims', 'low_risk_count', 
                      'medium_risk_count', 'high_risk_count', 'critical_risk_count')
        }),
        ('Generated Files', {
            'fields': ('output_csv_path', 'high_risk_csv_path', 'visualizations')
        }),
    )

@admin.register(Claim)
class ClaimAdmin(admin.ModelAdmin):
    list_display = ['claim_number', 'claimant_name', 'risk_level', 
                   'fraud_score', 'date_of_loss', 'analysis']
    list_filter = ['risk_level', 'date_of_loss', 'analysis']
    search_fields = ['claim_number', 'claimant_name']
    readonly_fields = ['red_flags']
    
    fieldsets = (
        ('Claim Information', {
            'fields': ('analysis', 'claim_number', 'claimant_name', 
                      'date_of_loss', 'injury_type', 'body_part')
        }),
        ('Fraud Analysis', {
            'fields': ('fraud_score', 'risk_level', 'red_flags', 'days_to_report')
        }),
        ('Financial', {
            'fields': ('claim_amount',)
        }),
    )