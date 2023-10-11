import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import plotly.io 
from io import BytesIO

st.set_page_config(layout="wide")

@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Calendared Study Visit')
    df = convert_to_datetime(df)
    return df

@st.cache_data
def convert_to_datetime(df):
    df[['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']] = df[['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']].apply(
        lambda col: pd.to_datetime(col.astype(str).str.strip(), format='%d/%m/%Y', errors='coerce'))
    return df

@st.cache_data
def reshape_dataframe(df):
    df_long = df.melt(id_vars='Study ID', value_vars=['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4'], var_name='Visit', value_name='Date')
    df_long = df_long.dropna(subset=['Date'])
    df_long = df_long.sort_values('Date').reset_index(drop=True)
    return df_long

@st.cache_data
def add_count_columns(df_long):
    df_long['Visits per Day'] = df_long.groupby('Date')['Study ID'].transform('count')
    df_long['Cumulative Count'] = np.arange(1, len(df_long) + 1)
    return df_long.sort_values('Date')

@st.cache_data
def add_day_of_week(df_long):
    df_long['Day of Week'] = df_long['Date'].dt.day_name()
    #st.dataframe(df_long)
    return df_long

@st.cache_data
def generate_trace(df_long, visit):
    color_map = {
        'Visit 1': '#3B5BA5',
        'Visit 2': '#c6d7eb',
        'Visit 3': '#F3B941',
        'Visit 4': '#B2456E'
    }
    df_subset = df_long[df_long['Visit'] == visit]
    df_subset['Visit Count'] = df_subset.groupby('Visit').cumcount() + 1

    return go.Scatter(
        x = df_subset['Date'],
        y = df_subset['Cumulative Count'],  # Plot 'Cumulative Count' instead of 'Jittered Count'
        mode = 'markers',
        name = visit,
        marker=dict(color=color_map.get(visit, 'black')),  # Set color here
        hovertext = [
            f"Study ID: {row['Study ID']}<br>" +
            f"Day of Week: {row['Day of Week']}<br>" +
            f"Date: {row['Date'].strftime('%d/%m/%y')}<br>" +
            f"Visit: {row['Visit']}<br>" +
            f"Total Visits on This Day: {row['Visits per Day']}<br>" +
            f"Trials Cumulative Count: {row['Cumulative Count']}<br>"+
            f"Count for {row['Visit']}: {row['Visit Count']}" # Add Cumulative Sum here
            for _, row in df_subset.iterrows()
        ]
    )


@st.cache_data
def plot_cumulative_trials(df):
    df_long = reshape_dataframe(df)
    df_long = add_count_columns(df_long)
    df_long = add_day_of_week(df_long)
    
     # Set the titles and labels
    fig.update_layout(
        title='Current of Cumulative Trials Conducted vs Date',
        xaxis_title='Date',
        yaxis_title='Cumulative Trials',
        autosize=True,
    )
    # Initialize the figure first
    fig = go.Figure(data=[generate_trace(df_long, visit) for visit in df_long['Visit'].unique()])

    # Then update the layout
    fig.update_layout(
        title='Current of Cumulative Trials Conducted vs Date',
        xaxis_title='Date',
        yaxis_title='Cumulative Trials',
        autosize=True
    )

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Actual data up-to-date: {current_time}")
    return fig

color_map = {
    'Visit 1': '#3B5BA5',
    'Visit 2': '#c6d7eb',
    'Visit 3': '#F3B941',
    'Visit 4': '#B2456E'
}

@st.cache_data
def plot_visit_status(df_long):
    # Count the number of completed visits for each visit type
    completed_counts = df_long[df_long['Date'].dt.date <= datetime.now().date()].groupby('Visit').size()
    
    # Prepare data for the plot
    visit_types = ['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']
    max_visits = 120
    completed = [completed_counts.get(visit, 0) for visit in visit_types]
    remaining = [max_visits - comp for comp in completed]
    
     # Create color lists based on color_map
    completed_colors = [color_map.get(visit, '#000000') for visit in visit_types]
    remaining_colors = ['#000000'] * len(visit_types)
    # Create the bar chart
    fig = go.Figure(data=[
        go.Bar(name='Completed', x=visit_types, y=completed, marker_color=completed_colors),
        go.Bar(name='Remaining', x=visit_types, y=remaining, marker_color=remaining_colors)
    ])
    
    
    # Customize the layout
    fig.update_layout(
        barmode='stack',
        title='Visit Status: Completed vs Remaining',
        xaxis_title='Visit Type',
        yaxis_title='Number of Visits',
        autosize=True
    )
    return fig

@st.cache_data
def plot_gender_age_table(df):
    try:
        # Reset the index if it's a multi-index DataFrame
        if isinstance(df.index, pd.MultiIndex):
            df.reset_index(inplace=True)
        
        # Check if 'Gender' and 'age-tier' columns exist
        if 'Gender' not in df.columns or 'age-tier' not in df.columns:
            st.warning("Columns 'Gender' and 'age-tier' must exist in the DataFrame.")
            return
        
        # Check for non-scalar values in 'Gender' and handle them
        if any(df['Gender'].apply(lambda x: isinstance(x, (list, dict, tuple)))):
            st.warning("The 'Gender' column contains non-scalar values which can't be grouped. Please handle these values first.")
            return
        
        # Filter the DataFrame to only include the relevant gender and age-tiers
        filtered_df = df[df['Gender'].isin(['Female', 'Male']) & df['age-tier'].isin(['40-50', '51-60'])]
        
        # Create a pivot table to count the number of occurrences for each gender and age-tier combination
        pivot_table = pd.pivot_table(filtered_df, values='Study ID', index=['Gender'], columns=['age-tier'], aggfunc='count', fill_value=0, margins=True, margins_name='Grand Total')

        # Reorder the columns and index to make sure they appear in the desired order
        pivot_table = pivot_table.reindex(columns=['40-50', '51-60', 'Grand Total'], index=['Female', 'Male', 'Grand Total'])

        # Initialize HTML string
        # Initialize HTML string
        html_string = """
        <table style="width:100%; max-width:400px; border:1px solid black; border-collapse:collapse; margin: auto;">
            <thead>
                <tr style="background-color:#3B5BA5; color:black;">
                    <th style="border:1px solid black; text-align:center;" rowspan="2">Gender</th>
                    <th style="border:1px solid black; text-align:center;" colspan="3">Age Tier (years old)</th>
                </tr>
                <tr style="background-color:#3B5BA5; color:black;">
                    <th style="border:1px solid black; text-align:center;">40-50</th>
                    <th style="border:1px solid black; text-align:center;">51-60</th>
                    <th style="border:1px solid black; text-align:center;">Grand Total</th>
                </tr>
            </thead>
            <tbody>
        """

        # Populate table rows
        for i, row in pivot_table.iterrows():
            html_string += f"<tr style='background-color:#c6d7eb; color:black;'>"
            html_string += f"<td style='border:1px solid black; text-align:center;'>{i}</td>"
            for col in ['40-50', '51-60', 'Grand Total']:
                html_string += f"<td style='border:1px solid black; text-align:center;'>{row[col]}</td>"
            html_string += "</tr>"
        
        return html_string

    except Exception as e:
        st.warning(f"An error occurred: {e}")
        
def generate_html(fig):
    return plotly.io.to_html(fig, include_plotlyjs='cdn', full_html=True)

def run_cumulative_trials_plot():
    st.title('ABLE Visits Progression')
    uploaded_file = st.sidebar.file_uploader("drop that dope here", type="xlsx")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        current_date = datetime.now().date()
        df_long = reshape_dataframe(df)
    
        data_filters = {
            "ABLE all Visits (cumu_sum), Current and Future projections": df_long,
            "Completed Visits": df_long[df_long['Date'].dt.date <= current_date],
            "Upcoming Visits": df_long[df_long['Date'].dt.date > current_date],
        }
        
        # Create columns to display the plots side by side
        cols = st.columns([1, 1, 1])
        with st.expander("Download Plots"):
            for i, (filter_name, filtered_data) in enumerate(data_filters.items()):

                # Generate the plot based on the filtered data
                filtered_data = add_count_columns(filtered_data)
                filtered_data = add_day_of_week(filtered_data)
                fig = go.Figure(data=[generate_trace(filtered_data, visit) for visit in filtered_data['Visit'].unique()])

                # Set up the plot layout
                fig.update_xaxes(dtick="M1", tickformat="%b\n%Y", title="Date")
                fig.update_layout(
                    title=f'{filter_name}', 
                    xaxis_title='Date', 
                    yaxis_title='Cumulative Trials', 
                    autosize=True,  # Set autosize to True to make the plot responsive
                )

                # Display the plot in the respective column
                cols[i].plotly_chart(fig, use_container_width=True)  # Set use_container_width to True to make the plot fill the column width

                # Create an expander for the data frame and display the data frame in the respective column
                with cols[i].expander("Show Data"):
                    st.dataframe(filtered_data)

                html_string = plotly.io.to_html(fig, full_html=True, include_plotlyjs='cdn')
                # Create a download button for the plot
                html_out = BytesIO(html_string.encode())
                download_button_label = f"Download {filter_name} Plot as HTML"
                st.download_button(
                    label=download_button_label,
                    data=html_out,
                    file_name=f'{filter_name}.html',
                    mime='text/html',
                )
            
        # Create new columns to display the plots side by side
        status_gender_cols = st.columns([1, 1])
        
        # Plotting the visit status
        with status_gender_cols[0]:
            st.title('Visit Status')
            st.caption('Following chart demonstrates the status of our Visit Status...')
            visit_status_fig = plot_visit_status(df_long)  # Assign the returned figure to a variable
            st.plotly_chart(visit_status_fig)  # Plot the figure

        # Plotting the gender and age-tier table
        with status_gender_cols[1]:
            st.title('Gender and Age-Tier distribution in the Actual Participant')
            st.caption('Following table shows the count of individuals by Gender and Age Tier...')
            html_string = plot_gender_age_table(df)
            st.markdown(html_string, unsafe_allow_html=True)
    

if __name__ == "__main__":
    run_cumulative_trials_plot()
