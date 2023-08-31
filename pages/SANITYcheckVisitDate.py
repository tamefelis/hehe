import pandas as pd
from datetime import timedelta
import streamlit as st
import numpy as np
import io
import base64

class DateValidator:
    instance_count = 0  # static variable to track the number of instances

    def __init__(self, uploaded_file):
        DateValidator.instance_count += 1
        xls = pd.ExcelFile(uploaded_file)
        default_sheet_index = xls.sheet_names.index('Calendared Study Visit') if 'Calendared Study Visit' in xls.sheet_names else 0
        self.sheet_name = st.selectbox('Choose a worksheet:', xls.sheet_names, index=default_sheet_index, key=f"sheet_name{DateValidator.instance_count}")
        self.df = xls.parse(self.sheet_name)  # Parsing the selected sheet into self.df
        if 'Start time' in self.df.columns:
            self.df['Start time'] = self.df['Start time'].astype(str)
        else:
                st.warning("")

        #self.df['Start time'] = self.df['Start time'].astype(str)
        for col in ['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']:
            self.df[col] = pd.to_datetime(self.df[col].str.strip(), format='%d/%m/%Y', errors='coerce')


        
    def validate_dates(self, dates):
        non_compliant_visits = []
        comments = []
        for i in range(3):
            if pd.isnull(dates[i]) or pd.isnull(dates[i + 1]):
                continue
            if not timedelta(days=84) <= (dates[i + 1] - dates[i]) <= timedelta(days=98):
                non_compliant_visits.extend([f'Visit {i + 1}', f'Visit {i + 2}'])
                comments.append(f'Difference between {dates[i]} ({f"Visit {i + 1}"}) and {dates[i + 1]} ({f"Visit {i + 2}"}) is not within ±7 days of 3 months')
        return non_compliant_visits, comments


    def process_dataframe(self):
        df_new = pd.DataFrame(columns=['Study ID', 'Non-compliant Column', 'Comments'])
        for idx, row in self.df.iterrows():
            dates = [row['Visit 1'], row['Visit 2'], row['Visit 3'], row['Visit 4']]
            non_compliant_visits, comments = self.validate_dates(dates)
            if non_compliant_visits:
                df_new.loc[len(df_new)] = {
                    'Study ID': row['Study ID'],
                    'Non-compliant Column': ' & '.join(set(non_compliant_visits)),
                    'Comments': ' | '.join(comments)
                }
        return df_new

    def filter_visits(self):
            # Reshape the DataFrame to a long format
        df_long = self.df.melt(id_vars='Study ID', value_vars=['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4'], var_name='Visit', value_name='Date')
        # Remove any rows with missing dates
        df_long = df_long.dropna(subset=['Date'])
        # Create a column for the number of visits each day
        df_long['Visits per Day'] = df_long.groupby('Date')['Study ID'].transform('count')
        # Sort the DataFrame by Date
        df_long = df_long.sort_values('Date')
        # Create a cumulative count column
        df_long['Cumulative Count'] = np.arange(1, len(df_long) + 1)
        # Create a column for day of the week
        df_long['Day of Week'] = df_long['Date'].dt.day_name()
        # Filter out the Date where Visits per Day > 2
        df_filtered = df_long[df_long['Visits per Day'] > 2]
        return df_filtered
    
def generate_csv(df):
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()
        b64_csv = base64.b64encode(csv_str.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64_csv}" download="result.csv">Download CSV File</a>'
        
        return href, csv_str

def main():
    st.title("ABLE Screening and Scheduling Checker")
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])

    if uploaded_file is not None:
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox('Choose a worksheet:', xls.sheet_names)
        df = xls.parse(sheet_name)

        validator = DateValidator(uploaded_file)  # Create an instance of DateValidator
        result_df = validator.process_dataframe() # Process the data

        st.header("Non-compliant Rows:")
        st.write(result_df)

        st.header('Screen out Date where visit is > 2')
        validator2 = DateValidator(uploaded_file)
        result2df = validator2.filter_visits()
        
        # Call the new filter_visits function with the DataFrame
        st.write(result2df)  # Display the result
        
        # Generate CSV for the second DataFrame
        href2, csv_str2 = generate_csv(result2df)
        st.markdown(href2, unsafe_allow_html=True)
        st.download_button("Download CSV for Screened-out Dates", csv_str2, file_name="screened_out_dates.csv", mime="text/csv")

if __name__ == '__main__':
    main()
