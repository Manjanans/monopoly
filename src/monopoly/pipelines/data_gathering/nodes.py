import pandas as pd
import gdown
from io import BytesIO
import ssl
import tempfile
from pathlib import Path

def extract_transform_data() -> pd.DataFrame:
    ssl._create_default_https_context = ssl._create_unverified_context
    file_id = '120szsMnVIMgze6B-KboYJVjfVBy5UwRN'
    download_url = f'https://drive.google.com/uc?id={file_id}'

    file_path = Path("data/01_raw/data.pq")  # Relative to the Kedro project root
    if file_path.exists():
        print(f"File already exists at {file_path}. Skipping download.")
        data = pd.read_parquet(file_path)
    else:
        print("File not found. Downloading...")
        # Download to memory (BytesIO object)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            gdown.download(download_url, temp_file.name, quiet=False)

            # Read the downloaded file as an Excel file
            data = pd.read_excel(temp_file.name, sheet_name='Transición de Negocio')
    
    return data

def basic_data(data:pd.DataFrame) -> pd.DataFrame:
    """
    This function takes the data and returns a dataframe with the following basic information:
        
        ID: ID of the client
        Sex: Sex of the client (0 for female, 1 for male)
        Age: Age of the client, from 9 to 104
        Num_CC: Number of Credit Cards of the client
        National: National credit of the client in Chilean Pesos
        International: International credit of the client in US Dolars
        Income: Income of the client in Chilean Pesos
        Num_Acc: Number of accounts of the client
        Mon_Act: How long, in months, the client has been with the bank

    Then, it transforms the following data:

        - Sex: 1 for male, 0 for female
        - Age: Into different groups:
            - 0: 0 to 25 years old
            - 1: 25 to 30 years old
            - 2: 30 to 40 years old
            - 3: 40 to 55 years old
            - 4: Over 55 years old

    Args:
        data: Dataframe with the data
    returns:
        Dataframe with the basic information
    """
    client_data = pd.DataFrame(data[['Id','Sexo','Edad','TC','CUPO_L1','CUPO_MX','Renta','Cuentas','Antiguedad']])
    client_data = client_data.rename(columns={
        'Id': 'ID',
        'Sexo': 'Sex',
        'Edad': 'Age',
        'TC': 'Num_CC',
        'CUPO_L1': 'National',
        'CUPO_MX': 'International',
        'Renta': 'Income',
        'Cuentas':'Num_Acc',
        'Antiguedad':'Mon_Act'
    })
    client_data['Sex'] = client_data['Sex'].map({'H': 1, 'M': 0})


    client_data.loc[(client_data['Age']<=25), 'Age'] = 0
    client_data.loc[((client_data['Age']>25) & (client_data['Age']<=30)), 'Age'] = 1
    client_data.loc[((client_data['Age']>30) & (client_data['Age']<=40)), 'Age'] = 2
    client_data.loc[((client_data['Age']>40) & (client_data['Age']<=55)), 'Age'] = 3
    client_data.loc[(client_data['Age']>55), 'Age'] = 4
    return client_data

def data_selection(data:pd.DataFrame)->pd.DataFrame:
    """
    This function takes the data and returns a dataframe with the following information:

        ID: ID of the client
        Int_Bill_CC_XX: Amount billed for international purchases by the client on credit card in month X
        Nac_Bill_CC_XX: Amount billed for domestic purchases by the client on credit card in month X
        Adv_Bill_DC_XX: Amount billed for cash advances by the client on debit card in month X
        Pur_Bill_DC_XX: Amount billed for purchases by the client on debit card in month X
        Act_Indi_CC_XX: Activity indicator in month X on the credit card
        Int_Acti_CC_XX: Activity indicator for international purchases in month X on the credit card
        Nac_Acti_CC_XX: Activity indicator for domestic purchases in month X on the credit card
        Num_Tran_CC_XX: Number of transactions made by the client on credit card in month X
        Int_Tran_CC_XX: Number of international purchase transactions made by the client on credit card in month X
        Nac_Tran_CC_XX: Number of domestic purchase transactions made by the client on credit card in month X
        Adv_Tran_DC_XX: Number of cash advance transactions made by the client on debit card in month X
        Pur_Tran_DC_XX: Number of purchase transactions made by the client on debit card in month X
    
    Args:
        data: Dataframe with the data
    Returns:
        Dataframe with the data selected
    """

    #Creation of the list to contain all the data
    all_transactions = []

    #For loop that goes from 1 to 12, to get all the months. Then, all the months are stored in a DataFrame in the list above.
    for month in range(1,13):
        X = month
        if X<10:   
            IBC = f'Int_Bill_CC_0{X}'
            NBC = f'Nac_Bill_CC_0{X}'
            ABD = f'Adv_Bill_DC_0{X}'
            PBD = f'Pur_Bill_DC_0{X}'
            AIC = f'Act_Indi_CC_0{X}'
            IAC = f'Int_Acti_CC_0{X}'
            NAC = f'Nac_Acti_CC_0{X}'
            NTC = f'Num_Tran_CC_0{X}'
            ITC = f'Int_Tran_CC_0{X}'
            NTCC = f'Nac_Tran_CC_0{X}'
            ATD = f'Adv_Tran_DC_0{X}'
            PTD = f'Pur_Tran_DC_0{X}'
            col1 = f'FacCI_T0{X}'
            col2 = f'FacCN_T0{X}'
            col3 = f'FacDebAtm_T0{X}'
            col4 = f'FacDebCom_T0{X}'
            col5 = f'FlgAct_T0{X}'
            col6 = f'FlgActCI_T0{X}'
            col7 = f'FlgActCN_T0{X}'
            col8 = f'Txs_T0{X}'
            col9 = f'TxsCI_T0{X}'
            col10 = f'TxsCN_T0{X}'
            col11 = f'TxsDebAtm_T0{X}'
            col12 = f'TxsDebCom_T0{X}'
        else:
            IBC = f'Int_Bill_CC_{X}'
            NBC = f'Nac_Bill_CC_{X}'
            ABD = f'Adv_Bill_DC_{X}'
            PBD = f'Pur_Bill_DC_{X}'
            AIC = f'Act_Indi_CC_{X}'
            IAC = f'Int_Acti_CC_{X}'
            NAC = f'Nac_Acti_CC_{X}'
            NTC = f'Num_Tran_CC_{X}'
            ITC = f'Int_Tran_CC_{X}'
            NTCC = f'Nac_Tran_CC_{X}'
            ATD = f'Adv_Tran_DC_{X}'
            PTD = f'Pur_Tran_DC_{X}'
            col1 = f'FacCI_T{X}'
            col2 = f'FacCN_T{X}'
            col3 = f'FacDebAtm_T{X}'
            col4 = f'FacDebCom_T{X}'
            col5 = f'FlgAct_T{X}'
            col6 = f'FlgActCI_T{X}'
            col7 = f'FlgActCN_T{X}'
            col8 = f'Txs_T{X}'
            col9 = f'TxsCI_T{X}'
            col10 = f'TxsCN_T{X}'
            col11 = f'TxsDebAtm_T{X}'
            col12 = f'TxsDebCom_T{X}'
        monthly_transactions = data[['Id',col1, col2, col3, col4, col5, col6,
                                    col7, col8, col9, col10, col11, col12]].rename(columns={
            'Id':'ID',
            col1: IBC,
            col2: NBC,
            col3: ABD,
            col4: PBD,
            col5: AIC,
            col6: IAC,
            col7: NAC,
            col8: NTC,
            col9: ITC,
            col10: NTCC,
            col11: ATD,
            col12: PTD
        })
        all_transactions.append(monthly_transactions)

    #First, it creates a dataset, with the first item on the list (first month).
    client_transactions = all_transactions[0]
    #For loop to merge the rest of the months
    for data_month in all_transactions[1:]:
        client_transactions = pd.merge(client_transactions,data_month, on='ID',how='outer')

    return client_transactions

def data_merge(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function does as following:
        1. It creates a dataset with the basic data of the client
        2. It creates a dataset with the data of the transactions
        3. It merges the two datasets

    Args:
        data (pd.DataFrame): The dataset of the clients

    Returns:
        pd.DataFrame: Client Data
        pd.DataFrame: Transactions Data
        pd.DataFrame: Merged Data
    """
    client_data = basic_data(data)
    client_transactions = data_selection(data)
    merged_data = pd.merge(client_data,client_transactions, on='ID',how='outer')
    return client_data,client_transactions,merged_data

def data_groups(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function does as following:

        1. Creates a dataset with the basic data of the client
        2. Creates a list, with the prefixes of each of the columns.
        3. For loop, to sum the columns that have the same prefix.
        4. Adds the new columns to the dataset.
        5. Returns the dataset

    The columns of the dataset are the following:

        ID: ID of the client
        Sex: Sex of the client (0 for female, 1 for male)
        Age: Into different groups:
            - 0: 0 to 25 years old
            - 1: 25 to 30 years old
            - 2: 30 to 40 years old
            - 3: 40 to 55 years old
            - 4: Over 55 years old
        Num_CC: Number of Credit Cards of the client
        National: National credit of the client in Chilean Pesos
        International: International credit of the client in US Dolars
        Income: Income of the client in Chilean Pesos
        Num_Acc: Number of accounts of the client
        Mon_Act: Number of active months of the client
        Total_Int_Bill_CC: Total amount of international bills of the client
        Total_Nac_Bill_CC: Total amount of national bills of the client
        Total_Adv_Bill_DC: Total amount of advance bills of the client
        Total_Pur_Bill_DC: Total amount of purchase bills of the client
        Total_Act_Indi_CC: Total amount of individual accounts of the client
        Total_Int_Acti_CC: Total amount of international activity of the client
        Total_Nac_Acti_CC: Total amount of national activity of the client
        Total_Num_Tran_CC: Total number of transactions of the client
        Total_Int_Tran_CC: Total amount of international transactions of the client
        Total_Nac_Tran_CC: Total amount of national transactions of the client
        Total_Adv_Tran_DC: Total amount of advance transactions of the client
        Total_Pur_Tran_DC: Total amount of purchase transactions of the client 

    Args:
        merged_data (pd.DataFrame): The merged dataset of the clients

    Returns:
        pd.DataFrame: Data grouped
    """
    groups = pd.DataFrame(merged_data[['ID', 'Sex', 'Age', 'Num_CC', 'National', 'International','Income','Num_Acc','Mon_Act']])
    prefixes = [
    'Int_Bill_CC_', 'Nac_Bill_CC_', 'Adv_Bill_DC_', 'Pur_Bill_DC_',
    'Act_Indi_CC_', 'Int_Acti_CC_', 'Nac_Acti_CC_', 'Num_Tran_CC_',
    'Int_Tran_CC_', 'Nac_Tran_CC_', 'Adv_Tran_DC_', 'Pur_Tran_DC_'
    ]
    for prefix in prefixes:
        columns_to_sum = [f'{prefix}{i:02}' for i in range(1, 13)]
        groups['Total_'+prefix[:11]] = merged_data[columns_to_sum].sum(axis=1)

    return groups

def more_groups(data:pd.DataFrame, merged_data:pd.DataFrame) -> pd.DataFrame:
    all_cols = []
    other_data = pd.DataFrame(data[['Id','Consumo','Hipotecario','Edad','Sexo']])
    other_data = other_data.rename(columns={
        'Id':'ID',
        'Consumo':'Consumer',
        'Hipotecario':'Mortgage',
        'Edad':'Age',
        'Sexo':'Sex'
    })
    for month in range(1,13):
        X = month
        if X<10:
            uso1 = f'Nac_Bought_CC_0{X}'
            uso2 = f'Nac_Advanc_CC_0{X}'
            uso3 = f'Int_Bought_CC_0{X}'
            col1 = f'UsoL1_T0{X}'
            col2 = f'UsoL2_T0{X}'
            col3 = f'UsoLI_T0{X}'
        else:
            uso1 = f'Nac_Bought_CC_{X}'
            uso2 = f'Nac_Advanc_CC_{X}'
            uso3 = f'Int_Bought_CC_{X}'
            col1 = f'UsoL1_T{X}'
            col2 = f'UsoL2_T{X}'
            col3 = f'UsoLI_T{X}'
        monthly_transactions = data[['Id',col1, col2, col3]].rename(columns={
            'Id':'ID',
            col1: uso1,
            col2: uso2,
            col3: uso3,
        })
        all_cols.append(monthly_transactions)

    #First, it creates a dataset, with the first item on the list (first month).
    client_transactions = all_cols[0]
    #For loop to merge the rest of the months
    for data_month in all_cols[1:]:
        client_transactions = pd.merge(client_transactions,data_month, on='ID',how='outer')
        
    prefixes = ['Nac_Bought_CC_', 'Nac_Advanc_CC_', 'Int_Bought_CC_']
    for prefix in prefixes:
        columns_to_sum = [f'{prefix}{i:02}' for i in range(1, 13)]
        other_data['Total_'+prefix[:12]] = client_transactions[columns_to_sum].sum(axis=1)
    
    return other_data