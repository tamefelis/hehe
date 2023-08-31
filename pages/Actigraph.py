import streamlit as st
import pandas as pd
import os
import logging

def process_data(file, sheet_name):
    # Read Excel from the uploaded file
    df = pd.read_excel(file, sheet_name=sheet_name)

    # Columns to preprocess
    visit_cols = ['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']

    # Preprocess date columns
    for col in visit_cols:
        df[col] = pd.to_datetime(df[col].astype(str).str.strip(), format='%d/%m/%Y', errors='coerce')

    # Create temporary DataFrame with Study ID
    temp_df = pd.DataFrame()
    temp_df['Study ID'] = df['Study ID']

    # Calculate min and max datetimes for each visit
    for i, col in enumerate(visit_cols, start=1):
        min_datetime_col = f'V{i}MinDatetime'
        max_datetime_col = f'V{i}MaxDatetime'
        temp_df[min_datetime_col] = df[col].dt.strftime('%Y-%m-%d') + 'T07:00:00'
        temp_df[max_datetime_col] = (df[col] + pd.Timedelta(days=8)).dt.strftime('%Y-%m-%d') + 'T23:59:00'

    return temp_df

def truncate_data(temp_df, csv_directory):
    for index, row in temp_df.iterrows():
        study_id = row['Study ID']
        file_name = f"{study_id}.epochsummarydata.csv"
        full_path = os.path.join(csv_directory, file_name)

        if os.path.exists(full_path):
            csv_df = pd.read_csv(full_path)
            # Check if the 'Subject' column contains the same Study ID
            if csv_df['Subject'].nunique() == 1 and csv_df['Subject'].iloc[0] == study_id:
                # Make sure the 'Timestamp' column is in datetime format
                csv_df['Timestamp'] = pd.to_datetime(csv_df['Timestamp'])

                # Iterate through visits and truncate data
                for i in range(1, 5):
                    min_datetime = pd.to_datetime(row[f'V{i}MinDatetime'])
                    max_datetime = pd.to_datetime(row[f'V{i}MaxDatetime'])
                    truncated_data = csv_df[(csv_df['Timestamp'] >= min_datetime) & (csv_df['Timestamp'] <= max_datetime)]

                    # Print the retained data (optional)
                    print(truncated_data)

                    # Create folder with the name of Study ID
                    study_id_folder = os.path.join(csv_directory, study_id)
                    os.makedirs(study_id_folder, exist_ok=True)

                    # Save the preprocessed CSV
                    new_file_name = f"{study_id}epochsummaryT.csv"
                    truncated_data.to_csv(os.path.join(study_id_folder, new_file_name), index=False)

                    logging.info(f'Data processed for Study ID: {study_id}, Visit: V{i}, File: {file_name}')
            else:
                logging.warning(f'Subject ID mismatch for Study ID: {study_id}, File: {file_name}')
        else:
            logging.warning(f'Missing data for Study ID: {study_id}, File: {file_name}')

    logging.info('Data processing completed.')

def main():
    st.title('Visit Datetime Processor')

    # Directory path for CSV files
    csv_directory = "C:\\Users\\yuenchl\\Desktop\\actigraph\\14082023"

    # File uploader for Excel file
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

    if uploaded_file is not None:
        # Get sheet names
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names

        # Select sheet
        selected_sheet = st.selectbox('Select Sheet', sheet_names)

        # Process the uploaded file
        processed_data = process_data(uploaded_file, selected_sheet)

        # Display the processed data
        st.write("Processed Visit Data:")
        st.dataframe(processed_data)

        # Call the truncate_data function with temp_df and CSV directory
        truncate_data(processed_data, csv_directory)
        # Link to download log file
        st.markdown("[Download log file](sandbox:/path/to/your/log/file/processing_log.log)")

if __name__ == "__main__":
    main()