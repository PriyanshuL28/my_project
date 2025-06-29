from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse, FileResponse
from django.contrib import messages
from django.views.generic import ListView, DetailView
from django.core.paginator import Paginator
from django.db.models import Q
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
import traceback

from .models import FraudAnalysis, Claim
from .forms import UploadFileForm
from .utils.fraud_detector import FraudDetector
from .utils.visualization import FraudVisualizer

def clean_currency_column(series):
    """Clean currency columns by removing $ and , symbols"""
    if series.dtype == 'object':
        # Remove currency symbols and commas
        series = series.astype(str).str.replace('$', '', regex=False)
        series = series.str.replace(',', '', regex=False)
        series = series.str.strip()
        # Convert to numeric, replacing errors with NaN
        series = pd.to_numeric(series, errors='coerce')
    return series

def index(request):
    """Home page view"""
    recent_analyses = FraudAnalysis.objects.all()[:5]
    
    # Get overall statistics
    total_analyses = FraudAnalysis.objects.count()
    total_claims = Claim.objects.count()
    high_risk_claims = Claim.objects.filter(risk_level__in=['High', 'Critical']).count()
    
    context = {
        'recent_analyses': recent_analyses,
        'total_analyses': total_analyses,
        'total_claims': total_claims,
        'high_risk_claims': high_risk_claims,
        'high_risk_percentage': (high_risk_claims / total_claims * 100) if total_claims > 0 else 0,
    }
    return render(request, 'fraud_detector/index.html', context)

def upload_file(request):
    """Handle file upload and processing"""
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            analysis = form.save(commit=False)
            if request.user.is_authenticated:
                analysis.user = request.user
            analysis.save()
            
            # Process the file
            try:
                result = process_fraud_analysis(analysis)
                if result['success']:
                    messages.success(request, 'File uploaded and analyzed successfully!')
                    return redirect('fraud_detector:dashboard', analysis_id=analysis.id)
                else:
                    messages.error(request, f'Error processing file: {result["error"]}')
                    analysis.delete()
            except Exception as e:
                messages.error(request, f'Error processing file: {str(e)}')
                print(f"Upload error: {traceback.format_exc()}")
                if 'analysis' in locals():
                    analysis.delete()
    else:
        form = UploadFileForm()
    
    return render(request, 'fraud_detector/upload.html', {'form': form})

def process_fraud_analysis(analysis):
    """Process the uploaded CSV file for fraud detection"""
    try:
        # Read the CSV file
        df = pd.read_csv(analysis.uploaded_file.path, thousands=',', low_memory=False)
        
        print(f"Loaded CSV with shape: {df.shape}")
        
        # List of columns that might contain currency/numeric data
        potential_numeric_columns = [
            'Claim Incurred - Total', 'Claim Incurred – Total',
            'Claim Paid - Total', 'Claim Paid – Total',
            'Claim Future Reserve - Total', 'Claim Future Reserve – Total',
            'Claim Incurred - Medical', 'Claim Incurred – Medical',
            'Claim Paid - Medical', 'Claim Paid – Medical',
            'Claim Future Reserve - Medical', 'Claim Future Reserve – Medical',
            'Claim Incurred - Ind/Loss', 'Claim Incurred – Ind/Loss',
            'Claim Paid - Ind/Loss', 'Claim Paid – Ind/Loss',
            'Claim Future Reserve - Ind/Loss', 'Claim Future Reserve – Ind/Loss',
            'Claim Incurred - Expense', 'Claim Incurred – Expense',
            'Claim Paid - Expense', 'Claim Paid – Expense',
            'Claim Future Reserve - Expense', 'Claim Future Reserve – Expense',
            'Claim Incurred - Legal', 'Claim Incurred – Legal',
            'Claim Paid - Legal', 'Claim Paid – Legal',
            'Claim Future Reserve - Legal', 'Claim Future Reserve – Legal',
            'Pre Injury AWW', 'Wage Base', 'Weekly Wage', 'Current Wage',
            'Deductible', 'Policy Deductible', 'Claim Amount',
            'Claim Recovery - Total', 'Claim Recovery – Total',
            'days_to_report', 'Event Time'
        ]
        
        # Clean numeric columns
        for col in potential_numeric_columns:
            if col in df.columns:
                df[col] = clean_currency_column(df[col])
                print(f"Cleaned column: {col}")
        
        # Initialize fraud detector
        detector = FraudDetector()
        
        # Detect fraud
        df_with_fraud = detector.detect_fraud(df)
        
        print(f"Fraud detection completed. Risk distribution: {df_with_fraud['risk_level'].value_counts().to_dict()}")
        
        # Generate output files
        output_dir = os.path.join('media', 'outputs', f'analysis_{analysis.id}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed CSV
        output_csv_path = os.path.join(output_dir, 'fraud_analysis_results.csv')
        df_with_fraud.to_csv(output_csv_path, index=False)
        analysis.output_csv_path = output_csv_path.replace('media/', '')
        
        # Save high-risk claims CSV
        high_risk_df = df_with_fraud[df_with_fraud['risk_level'].isin(['High', 'Critical'])]
        if not high_risk_df.empty:
            high_risk_csv_path = os.path.join(output_dir, 'high_risk_claims.csv')
            high_risk_df.to_csv(high_risk_csv_path, index=False)
            analysis.high_risk_csv_path = high_risk_csv_path.replace('media/', '')
        
        # Generate visualizations
        viz_dir = os.path.join('media', 'visualizations', f'analysis_{analysis.id}')
        visualizer = FraudVisualizer(viz_dir)
        visualizations = visualizer.generate_all_visualizations(df_with_fraud)
        
        # Store visualization paths
        viz_paths = {}
        for key, filename in visualizations.items():
            viz_paths[key] = os.path.join('visualizations', f'analysis_{analysis.id}', filename)
        analysis.visualizations = viz_paths
        
        # Update summary statistics
        risk_counts = df_with_fraud['risk_level'].value_counts()
        analysis.total_claims = len(df_with_fraud)
        analysis.low_risk_count = int(risk_counts.get('Low', 0))
        analysis.medium_risk_count = int(risk_counts.get('Medium', 0))
        analysis.high_risk_count = int(risk_counts.get('High', 0))
        analysis.critical_risk_count = int(risk_counts.get('Critical', 0))
        analysis.processed_at = datetime.now()
        analysis.save()
        
        # Save individual claims to database
        save_claims_to_db(analysis, df_with_fraud)
        
        return {'success': True}
        
    except Exception as e:
        print(f"Error in process_fraud_analysis: {str(e)}")
        print(traceback.format_exc())
        return {'success': False, 'error': str(e)}

def save_claims_to_db(analysis, df):
    """Save individual claims to the database"""
    claims_to_create = []
    
    for _, row in df.iterrows():
        # Handle claim amount - try multiple possible column names
        claim_amount = None
        amount_columns = [
            'Claim Incurred - Total', 'Claim Incurred – Total',
            'Claim Paid - Total', 'Claim Paid – Total'
        ]
        
        for col in amount_columns:
            if col in row and pd.notna(row[col]):
                try:
                    claim_amount = float(row[col])
                    break
                except (ValueError, TypeError):
                    continue
        
        # Safely get days_to_report
        days_to_report = None
        if 'days_to_report' in row and pd.notna(row['days_to_report']):
            try:
                days_to_report = int(float(row['days_to_report']))
            except:
                pass
        
        claim = Claim(
            analysis=analysis,
            claim_number=str(row.get('Claim Number', ''))[:100],  # Ensure it fits in CharField
            claimant_name=str(row.get('Claimant Full Name', ''))[:200],
            date_of_loss=pd.to_datetime(row.get('Date of Loss'), errors='coerce'),
            injury_type=str(row.get('Injury Type Description', ''))[:200],
            body_part=str(row.get('Target/Part of Body Description', ''))[:200],
            fraud_score=float(row.get('fraud_score', 0)),
            risk_level=str(row.get('risk_level', 'Low')),
            red_flags=row.get('red_flags', []),
            days_to_report=days_to_report,
            claim_amount=claim_amount
        )
        claims_to_create.append(claim)
    
    # Bulk create claims
    Claim.objects.bulk_create(claims_to_create, batch_size=500)
    print(f"Created {len(claims_to_create)} claim records")

def analyze_patterns(analysis):
    """Analyze fraud patterns from claims data"""
    claims = analysis.claims.all()
    
    # Define pattern categories
    timing_patterns = [
        'Weekend injury', 'Monday morning injury', 'Near birthday', 
        'Near holiday', 'New employee (<30 days)', 'New employee (<90 days)',
        'End of month claim', 'Summer claim', 'Friday afternoon injury',
        'Claim shortly before termination', 'Unusual time of injury',
        'Relatively new employee (<90 days)'
    ]
    
    behavioral_patterns = [
        'Multiple claims from same person', 'Attorney involved immediately',
        'Soft tissue injury', 'Suspicious body part injured',
        'Pushing for quick settlement', 'Avoiding recommended treatment',
        'Unusually long treatment', 'Pattern of suspicious claims'
    ]
    
    reporting_patterns = [
        'Delayed reporting (>30 days)', 'No witness contacted',
        'High claim rate location', 'Unusual time of injury'
    ]
    
    # Count patterns
    pattern_counts = {
        'timing': {},
        'behavioral': {},
        'reporting': {}
    }
    
    # Count claims with each pattern type
    timing_claims_set = set()
    behavioral_claims_set = set()
    reporting_claims_set = set()
    
    # Analyze all claims for patterns
    for claim in claims:
        has_timing = False
        has_behavioral = False
        has_reporting = False
        
        if claim.red_flags:
            for flag in claim.red_flags:
                # Remove category prefix if present
                clean_flag = flag.replace('[TIMING] ', '').replace('[BEHAVIORAL] ', '').replace('[REPORTING] ', '').replace('[INJURY] ', '')
                
                # Categorize the flag
                if any(pattern in clean_flag for pattern in timing_patterns):
                    pattern_counts['timing'][clean_flag] = pattern_counts['timing'].get(clean_flag, 0) + 1
                    has_timing = True
                elif any(pattern in clean_flag for pattern in behavioral_patterns):
                    pattern_counts['behavioral'][clean_flag] = pattern_counts['behavioral'].get(clean_flag, 0) + 1
                    has_behavioral = True
                elif any(pattern in clean_flag for pattern in reporting_patterns):
                    pattern_counts['reporting'][clean_flag] = pattern_counts['reporting'].get(clean_flag, 0) + 1
                    has_reporting = True
        
        # Track unique claims with each pattern type
        if has_timing:
            timing_claims_set.add(claim.id)
        if has_behavioral:
            behavioral_claims_set.add(claim.id)
        if has_reporting:
            reporting_claims_set.add(claim.id)
    
    # Format pattern data for template
    def format_patterns(pattern_dict):
        return sorted([
            {'name': name, 'count': count} 
            for name, count in pattern_dict.items()
        ], key=lambda x: x['count'], reverse=True)
    
    # Get all indicators sorted by frequency
    all_indicators = []
    for category_patterns in pattern_counts.values():
        for name, count in category_patterns.items():
            all_indicators.append({'name': name, 'count': count})
    all_indicators.sort(key=lambda x: x['count'], reverse=True)
    
    # Calculate percentages based on unique claims
    total_claims = analysis.total_claims
    timing_claims = len(timing_claims_set)
    behavioral_claims = len(behavioral_claims_set)
    reporting_claims = len(reporting_claims_set)
    
    pattern_analysis = {
        'timing_patterns': format_patterns(pattern_counts['timing']),
        'behavioral_patterns': format_patterns(pattern_counts['behavioral']),
        'reporting_patterns': format_patterns(pattern_counts['reporting']),
        'timing_count': sum(pattern_counts['timing'].values()),
        'behavioral_count': sum(pattern_counts['behavioral'].values()),
        'reporting_count': sum(pattern_counts['reporting'].values()),
        'timing_percentage': (timing_claims / total_claims * 100) if total_claims > 0 else 0,
        'behavioral_percentage': (behavioral_claims / total_claims * 100) if total_claims > 0 else 0,
        'reporting_percentage': (reporting_claims / total_claims * 100) if total_claims > 0 else 0,
        'top_indicators': all_indicators
    }
    
    return pattern_analysis

def dashboard(request, analysis_id):
    """Display analysis dashboard with dynamic pattern analysis"""
    analysis = get_object_or_404(FraudAnalysis, id=analysis_id)
    
    # Get summary statistics
    summary_stats = {
        'total_claims': analysis.total_claims,
        'risk_distribution': {
            'Low': analysis.low_risk_count,
            'Medium': analysis.medium_risk_count,
            'High': analysis.high_risk_count,
            'Critical': analysis.critical_risk_count,
        },
        'high_risk_percentage': ((analysis.high_risk_count + analysis.critical_risk_count) / analysis.total_claims * 100) if analysis.total_claims > 0 else 0,
    }
    
    # Get top high-risk claims
    top_claims = analysis.claims.filter(risk_level__in=['High', 'Critical']).order_by('-fraud_score')[:10]
    
    # Get pattern analysis
    pattern_analysis = analyze_patterns(analysis)
    
    context = {
        'analysis': analysis,
        'summary_stats': summary_stats,
        'top_claims': top_claims,
        'visualizations': analysis.visualizations,
        'pattern_analysis': pattern_analysis,
    }
    
    return render(request, 'fraud_detector/dashboard.html', context)

def claims_table(request, analysis_id):
    """Display all claims in a searchable table"""
    analysis = get_object_or_404(FraudAnalysis, id=analysis_id)
    claims = analysis.claims.all()
    
    # Search functionality
    search_query = request.GET.get('search', '')
    if search_query:
        claims = claims.filter(
            Q(claim_number__icontains=search_query) |
            Q(claimant_name__icontains=search_query) |
            Q(injury_type__icontains=search_query) |
            Q(body_part__icontains=search_query)
        )
    
    # Filter by risk level
    risk_filter = request.GET.get('risk_level', '')
    if risk_filter:
        claims = claims.filter(risk_level=risk_filter)
    
    # Sorting
    sort_by = request.GET.get('sort', '-fraud_score')
    claims = claims.order_by(sort_by)
    
    # Pagination
    paginator = Paginator(claims, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'analysis': analysis,
        'page_obj': page_obj,
        'search_query': search_query,
        'risk_filter': risk_filter,
        'sort_by': sort_by,
    }
    
    return render(request, 'fraud_detector/claims_table.html', context)

def high_risk_claims(request, analysis_id):
    """Display only high and critical risk claims"""
    analysis = get_object_or_404(FraudAnalysis, id=analysis_id)
    claims = analysis.claims.filter(risk_level__in=['High', 'Critical']).order_by('-fraud_score')
    
    # Pagination
    paginator = Paginator(claims, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'analysis': analysis,
        'page_obj': page_obj,
    }
    
    return render(request, 'fraud_detector/high_risk_claims.html', context)

def download_file(request, analysis_id, file_type):
    """Download generated files"""
    analysis = get_object_or_404(FraudAnalysis, id=analysis_id)
    
    if file_type == 'full':
        file_path = os.path.join('media', analysis.output_csv_path)
        filename = 'fraud_analysis_results.csv'
    elif file_type == 'high_risk':
        file_path = os.path.join('media', analysis.high_risk_csv_path)
        filename = 'high_risk_claims.csv'
    else:
        return HttpResponse('Invalid file type', status=400)
    
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=filename)
    else:
        return HttpResponse('File not found', status=404)

def visualization_view(request, analysis_id, viz_type):
    """View individual visualization"""
    analysis = get_object_or_404(FraudAnalysis, id=analysis_id)
    
    if viz_type in analysis.visualizations:
        viz_path = os.path.join('media', analysis.visualizations[viz_type])
        if os.path.exists(viz_path):
            return FileResponse(open(viz_path, 'rb'), content_type='image/png')
    
    return HttpResponse('Visualization not found', status=404)

def analysis_history(request):
    """View all past analyses"""
    analyses = FraudAnalysis.objects.all()
    
    if request.user.is_authenticated:
        # Option to filter by user
        user_only = request.GET.get('user_only', 'false') == 'true'
        if user_only:
            analyses = analyses.filter(user=request.user)
    
    paginator = Paginator(analyses, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
    }
    
    return render(request, 'fraud_detector/analysis_history.html', context)