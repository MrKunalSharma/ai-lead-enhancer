# src/ui/streamlit_app.py
# Part 1: Imports and Configuration (SIMPLIFIED)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
from pathlib import Path

# Import from local module
try:
    from lead_modules import LeadScorer, TextAnalyzer, DataEnricher, MODEL_CONFIG, SCORING_CONFIG
except ImportError:
    # If that fails, try adding current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    from lead_modules import LeadScorer, TextAnalyzer, DataEnricher, MODEL_CONFIG, SCORING_CONFIG

# Page configuration - using correct parameter names
st.set_page_config(
    page_title="AI Lead Enhancer - Caprae Capital",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stAlert {
        background-color: #EBF5FF;
        border: 1px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_leads' not in st.session_state:
    st.session_state.processed_leads = None
if 'lead_scorer' not in st.session_state:
    st.session_state.lead_scorer = LeadScorer()
if 'text_analyzer' not in st.session_state:
    st.session_state.text_analyzer = TextAnalyzer()
if 'data_enricher' not in st.session_state:
    st.session_state.data_enricher = DataEnricher()

# Header
st.markdown('<h1 class="main-header">🎯 AI Lead Enhancer</h1>', unsafe_allow_html=True)
st.markdown("### Transform Your Leads into Actionable Intelligence")

# Part 2: Sidebar Configuration

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100?text=Caprae+Capital", width=300)
    st.markdown("---")
    st.markdown("### 🔧 Configuration")
    
    # Model selection
    model_type = st.selectbox(
        "Select Scoring Model",
        ["gradient_boosting", "random_forest", "neural_network"]
    )
    
    # Threshold settings
    score_threshold = st.slider(
        "Minimum Lead Score",
        min_value=0,
        max_value=100,
        value=50,
        step=5
    )
    
    # Enrichment options
    st.markdown("### 🔍 Enrichment Options")
    enrich_company = st.checkbox("Company Information", value=True)
    enrich_social = st.checkbox("Social Media Profiles", value=True)
    enrich_tech = st.checkbox("Technology Stack", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    if st.session_state.processed_leads is not None:
        total_leads = len(st.session_state.processed_leads)
        high_quality = len(st.session_state.processed_leads[
            st.session_state.processed_leads['lead_score'] >= score_threshold
        ])
        st.metric("Total Leads", total_leads)
        st.metric("High Quality Leads", high_quality)
        st.metric("Conversion Rate", f"{(high_quality/total_leads)*100:.1f}%")

# Main content area - Tab setup
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📊 Analytics", "🎯 Lead Details", "📈 Reports"])
# Part 3: Upload & Process Tab

with tab1:
    st.markdown("### Upload Your Lead Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file containing your lead data"
        )
        
        if uploaded_file is not None:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Successfully loaded {len(df)} leads")
            
            # Show data preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(), use_container_width=True)
            
            # Process button
            if st.button("🚀 Process Leads", type="primary", use_container_width=True):
                with st.spinner("Processing leads..."):
                    progress_bar = st.progress(0)
                    
                    # Process each lead
                    processed_data = []
                    for idx, row in df.iterrows():
                        # Update progress
                        progress_bar.progress((idx + 1) / len(df))
                        
                        # Score the lead
                        lead_data = row.to_dict()
                        score = st.session_state.lead_scorer.score_lead(lead_data)
                        lead_data['lead_score'] = score
                        
                        # Analyze text if description exists
                        if 'description' in lead_data and pd.notna(lead_data['description']):
                            sentiment = st.session_state.text_analyzer.analyze_sentiment(
                                lead_data['description']
                            )
                            topics = st.session_state.text_analyzer.extract_topics(
                                lead_data['description']
                            )
                            lead_data['sentiment_score'] = sentiment['compound']
                            lead_data['topics'] = ', '.join(topics) if topics else 'N/A'
                        
                        # Enrich data
                        if enrich_company or enrich_social or enrich_tech:
                            enriched = st.session_state.data_enricher.enrich_lead(lead_data)
                            lead_data.update(enriched)
                        
                        processed_data.append(lead_data)
                    
                    # Store processed data
                    st.session_state.processed_leads = pd.DataFrame(processed_data)
                    st.success(f"✅ Successfully processed {len(processed_data)} leads!")
    
    with col2:
        st.markdown("### 📋 Upload Guidelines")
        st.info("""
        **Required columns:**
        - company_name
        - contact_name
        - email
        - industry
        
        **Optional columns:**
        - description
        - revenue
        - employee_count
        - location
        """)
        
        # Download sample data
        if st.button("📥 Download Sample Data"):
            sample_data = pd.DataFrame({
                'company_name': ['Tech Corp', 'Data Systems', 'AI Solutions'],
                'contact_name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
                'email': ['john@techcorp.com', 'jane@datasystems.com', 'bob@aisolutions.com'],
                'industry': ['Technology', 'Software', 'AI/ML'],
                'description': [
                    'Leading provider of cloud solutions',
                    'Enterprise data management platform',
                    'Cutting-edge AI research company'
                ]
            })
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="Download Sample CSV",
                data=csv,
                file_name="sample_leads.csv",
                mime="text/csv"
            )
# Part 4: Analytics Tab

with tab2:
    st.markdown("### 📊 Lead Analytics Dashboard")
    
    if st.session_state.processed_leads is not None:
        df = st.session_state.processed_leads
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_score = df['lead_score'].mean()
            st.metric("Average Lead Score", f"{avg_score:.1f}")
        with col2:
            top_leads = len(df[df['lead_score'] >= 80])
            st.metric("Top Tier Leads", top_leads)
        with col3:
            if 'sentiment_score' in df.columns:
                avg_sentiment = df['sentiment_score'].mean()
                st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
        with col4:
            unique_industries = df['industry'].nunique() if 'industry' in df.columns else 0
            st.metric("Industries", unique_industries)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution
            fig_scores = px.histogram(
                df, 
                x='lead_score', 
                nbins=20,
                title="Lead Score Distribution",
                labels={'lead_score': 'Lead Score', 'count': 'Number of Leads'}
            )
            fig_scores.update_layout(showlegend=False)
            st.plotly_chart(fig_scores, use_container_width=True)
        
        with col2:
            # Industry breakdown
            if 'industry' in df.columns:
                industry_scores = df.groupby('industry')['lead_score'].mean().sort_values(ascending=True)
                fig_industry = px.bar(
                    x=industry_scores.values,
                    y=industry_scores.index,
                    orientation='h',
                    title="Average Score by Industry",
                    labels={'x': 'Average Lead Score', 'y': 'Industry'}
                )
                st.plotly_chart(fig_industry, use_container_width=True)
        
        # Sentiment analysis
        if 'sentiment_score' in df.columns:
            st.markdown("### 😊 Sentiment Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment vs Score scatter
                fig_scatter = px.scatter(
                    df,
                    x='sentiment_score',
                    y='lead_score',
                    title="Sentiment vs Lead Score",
                    labels={'sentiment_score': 'Sentiment Score', 'lead_score': 'Lead Score'},
                    color='lead_score',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # Topic word cloud placeholder
                st.info("📊 Topic Analysis")
                if 'topics' in df.columns:
                    all_topics = ', '.join(df['topics'].dropna().tolist()).split(', ')
                    topic_counts = pd.Series(all_topics).value_counts().head(10)
                    fig_topics = px.bar(
                        x=topic_counts.values,
                        y=topic_counts.index,
                        orientation='h',
                        title="Top Topics Mentioned",
                        labels={'x': 'Frequency', 'y': 'Topic'}
                    )
                    st.plotly_chart(fig_topics, use_container_width=True)
    else:
        st.info("📤 Please upload and process lead data to view analytics")
# Part 5: Lead Details Tab

with tab3:
    st.markdown("### 🎯 Lead Details")
    
    if st.session_state.processed_leads is not None:
        df = st.session_state.processed_leads
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            score_filter = st.slider(
                "Filter by Score Range",
                min_value=0,
                max_value=100,
                value=(0, 100),
                step=5
            )
        with col2:
            if 'industry' in df.columns:
                industries = ['All'] + df['industry'].unique().tolist()
                selected_industry = st.selectbox("Filter by Industry", industries)
            else:
                selected_industry = 'All'
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ['lead_score', 'company_name', 'sentiment_score']
                if 'sentiment_score' in df.columns else ['lead_score', 'company_name']
            )
        
        # Apply filters
        filtered_df = df[
            (df['lead_score'] >= score_filter[0]) & 
            (df['lead_score'] <= score_filter[1])
        ]
        
        if 'industry' in df.columns and selected_industry != 'All':
            filtered_df = filtered_df[filtered_df['industry'] == selected_industry]
        
        # Sort
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
        
        # Display leads
        st.markdown(f"### Showing {len(filtered_df)} leads")
        
        for idx, lead in filtered_df.iterrows():
            with st.expander(f"🏢 {lead.get('company_name', 'Unknown')} - Score: {lead['lead_score']:.1f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Contact Information**")
                    st.write(f"👤 {lead.get('contact_name', 'N/A')}")
                    st.write(f"📧 {lead.get('email', 'N/A')}")
                    st.write(f"🏭 {lead.get('industry', 'N/A')}")
                    
                    if 'description' in lead and pd.notna(lead['description']):
                        st.markdown("**Description**")
                        st.write(lead['description'])
                
                with col2:
                    st.markdown("**Analytics**")
                    st.write(f"🎯 Lead Score: {lead['lead_score']:.1f}")
                    if 'sentiment_score' in lead and pd.notna(lead['sentiment_score']):
                        st.write(f"😊 Sentiment: {lead['sentiment_score']:.2f}")
                    if 'topics' in lead and pd.notna(lead['topics']):
                        st.write(f"🏷️ Topics: {lead['topics']}")
                    
                    # Enriched data
                    if 'linkedin_url' in lead and pd.notna(lead['linkedin_url']):
                        st.markdown("**Enriched Data**")
                        st.write(f"🔗 [LinkedIn Profile]({lead['linkedin_url']})")
                    if 'technologies' in lead and pd.notna(lead['technologies']):
                        st.write(f"💻 Tech Stack: {lead['technologies']}")
    else:
        st.info("📤 Please upload and process lead data to view details")
# Part 6: Reports Tab

with tab4:
    st.markdown("### 📈 Export Reports")
    
    if st.session_state.processed_leads is not None:
        df = st.session_state.processed_leads
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Report Options")
            report_type = st.selectbox(
                "Select Report Type",
                ["Full Lead Report", "High Score Leads", "Executive Summary"]
            )
            
            include_enriched = st.checkbox("Include Enriched Data", value=True)
            include_analytics = st.checkbox("Include Analytics", value=True)
            
            # Date range filter
            st.markdown("#### 📅 Date Range")
            use_date_filter = st.checkbox("Filter by date processed")
            
        with col2:
            st.markdown("#### 📋 Export Format")
            export_format = st.radio(
                "Select format",
                ["CSV", "Excel", "PDF Summary"]
            )
            
            # Generate report
            if st.button("📥 Generate Report", type="primary", use_container_width=True):
                with st.spinner("Generating report..."):
                    # Filter data based on report type
                    if report_type == "High Score Leads":
                        report_df = df[df['lead_score'] >= score_threshold]
                    else:
                        report_df = df.copy()
                    
                    # Remove columns based on options
                    if not include_enriched:
                        enriched_cols = ['linkedin_url', 'twitter_url', 'technologies', 
                                       'company_size', 'founded_year']
                        report_df = report_df.drop(columns=[col for col in enriched_cols 
                                                          if col in report_df.columns])
                    
                    if not include_analytics:
                        analytics_cols = ['sentiment_score', 'topics']
                        report_df = report_df.drop(columns=[col for col in analytics_cols 
                                                          if col in report_df.columns])
                    
                    # Export based on format
                    if export_format == "CSV":
                        csv = report_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV Report",
                            data=csv,
                            file_name=f"lead_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    elif export_format == "Excel":
                        # Create Excel file in memory
                        from io import BytesIO
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            report_df.to_excel(writer, sheet_name='Lead Report', index=False)
                            
                            # Add summary sheet if Executive Summary
                            if report_type == "Executive Summary":
                                summary_data = {
                                    'Metric': ['Total Leads', 'Average Score', 'High Quality Leads', 
                                              'Top Industry', 'Avg Sentiment'],
                                    'Value': [
                                        len(df),
                                        f"{df['lead_score'].mean():.1f}",
                                        len(df[df['lead_score'] >= score_threshold]),
                                        df['industry'].mode()[0] if 'industry' in df.columns else 'N/A',
                                        f"{df['sentiment_score'].mean():.2f}" if 'sentiment_score' in df.columns else 'N/A'
                                    ]
                                }
                                summary_df = pd.DataFrame(summary_data)
                                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                        
                        excel_data = output.getvalue()
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=excel_data,
                            file_name=f"lead_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    elif export_format == "PDF Summary":
                        st.info("PDF generation requires additional libraries. Showing summary preview:")
                        
                        # Display summary
                        st.markdown("### Executive Summary")
                        st.markdown(f"""
                        **Report Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                        
                        **Key Metrics:**
                        - Total Leads Analyzed: {len(df)}
                        - Average Lead Score: {df['lead_score'].mean():.1f}
                        - High Quality Leads (≥{score_threshold}): {len(df[df['lead_score'] >= score_threshold])}
                        - Conversion Potential: {(len(df[df['lead_score'] >= score_threshold]) / len(df) * 100):.1f}%
                        
                        **Top Industries:**
                        """)
                        
                        if 'industry' in df.columns:
                            top_industries = df['industry'].value_counts().head(5)
                            for industry, count in top_industries.items():
                                st.write(f"- {industry}: {count} leads")
        
        # Report preview
        st.markdown("### 📋 Report Preview")
        if report_type == "High Score Leads":
            preview_df = df[df['lead_score'] >= score_threshold].head(10)
        else:
            preview_df = df.head(10)
        
        st.dataframe(preview_df, use_container_width=True)
        
    else:
        st.info("📤 Please upload and process lead data to generate reports")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; padding: 2rem;'>
        <p>AI Lead Enhancer - Powered by Advanced AI | Built for Caprae Capital AI-Readiness Challenge</p>
        <p>© 2024 - Transforming Business Development with Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True
)