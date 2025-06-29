from django.urls import path
from . import views

app_name = 'fraud_detector'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_file, name='upload'),
    path('dashboard/<int:analysis_id>/', views.dashboard, name='dashboard'),
    path('claims/<int:analysis_id>/', views.claims_table, name='claims_table'),
    path('high-risk/<int:analysis_id>/', views.high_risk_claims, name='high_risk_claims'),
    path('download/<int:analysis_id>/<str:file_type>/', views.download_file, name='download_file'),
    path('visualization/<int:analysis_id>/<str:viz_type>/', views.visualization_view, name='visualization'),
    path('history/', views.analysis_history, name='analysis_history'),
]