import os
from sqlalchemy import create_engine, text
import logging

# Optional: Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SQL_QUERY_TEMPLATE = '''
WITH CTE AS (
    SELECT DISTINCT 
        CCFarmerIssueToken.TokenID,
        CCFarmerIssueToken.TokenNumber,
        CCFarmerIssueToken.FarmerMasterLoginId,
        CCFarmerIssueToken.CategoryID,
        CCFarmerIssueToken.IssueDescription,
        CCFarmerIssueToken.Priority,
        CCFarmerIssueToken.Status,
        ADMasterLogin.FirstName + ' ' + ADMasterLogin.LastName as FarmerName,
        ADMasterLogin.MobileNo AS FarmerMobileNo,
        CCCallLog.CallID,
        CCCallLog.CallUniqueID,
        CCCallLog.IsAgenticAI,
        CCCallLog.InteractionDate,
        CCCallLog.InteractionType,
        CCCallLog.CallStartTime,
        CCCallLog.CallEndTime,
        CCCallLog.CallDuration,
        CCCallLog.CallStatus,
        CCCallLog.CallRecordingURL,
        CCCallLog.Notes,
        CCCallLog.CallRating,
        CCAgronomist.AgronomistId,
        CCAgronomist.AgronomistMasterLoginId,
        ROW_NUMBER() OVER (PARTITION BY CCFarmerIssueToken.TokenID ORDER BY CCCallLog.CallId DESC) AS rn
    FROM CCCallLog 
    INNER JOIN CCFarmerIssueToken ON CCCallLog.TokenID = CCFarmerIssueToken.TokenID 
    INNER JOIN CCAgronomist ON CCCallLog.AgronomistID = CCAgronomist.AgronomistId 
    INNER JOIN ADMasterLogin ON CCFarmerIssueToken.FarmerMasterLoginId = ADMasterLogin.MasterLoginId 
    INNER JOIN ADMasterLogin AS ADMasterLogin_1 ON CCAgronomist.AgronomistMasterLoginId = ADMasterLogin_1.MasterLoginId
    WHERE CCAgronomist.AgronomistMasterLoginId = :login_id
      AND CCFarmerIssueToken.Status IN ('Assigned','InProgress') 
      AND CCCallLog.CallStatus IN ('Connected','Transferred')
      AND CAST(CCCallLog.InteractionDate AS DATE) = CAST(dbo.CloudDate(1) AS DATE)
      AND CCCallLog.InteractionType = 'Call'
)
SELECT * FROM CTE WHERE rn = 1 ORDER BY InteractionDate DESC, CallStartTime DESC
'''
def create_db_engine():
    return create_engine(
        f"mssql+pymssql://{os.getenv('SQL_USER')}:{os.getenv('SQL_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('SQL_DB')}",
        echo=False,
        pool_pre_ping=True
    )

def get_mongo_connection_string():
    """Get MongoDB connection string from settings."""
    return (
        f"{os.getenv('MONGO_URI')}"
    )

async def get_user_by_phone(phone_number):
    """
    Retrieve user details from the database based on phone number.
    Processes the phone number by trimming to last 10 digits.

    Args:
        phone_number (str or any): The phone number to search for

    Returns:
        dict or None: User details if found, else None
    """
    engine = None

    try:
        if not isinstance(phone_number, str):
            phone_number = str(phone_number)

        # Extract last 10 digits
        processed_phone = phone_number[-10:] if len(phone_number) >= 10 else phone_number
        logger.info(f"Looking up user with processed phone: {processed_phone}")

        engine = create_db_engine()
        query = text("""
            SELECT * FROM ADMasterLogin 
            WHERE MobileNo = :phone_number
        """)

        with engine.connect() as connection:
            result = connection.execute(query, {"phone_number": processed_phone})
            row = result.fetchone()

            if not row:
                logger.warning(f"No user found with phone number: {processed_phone}")
                return None

            user_data = dict(zip(result.keys(), row))
            return user_data

    except Exception as e:
        logger.error(f"Error retrieving user by phone: {e}", exc_info=True)
        return None

    finally:
        if engine:
            engine.dispose()

async def get_farm(master_login_id):
    engine = create_db_engine()
    try:
        with engine.connect() as connection:
            query = text("""
                DECLARE @return_value int;

                EXEC @return_value = [dbo].[SP_AMMasterFarm]
                    @MasterFarmId = NULL,
                    @FarmTitle = NULL,
                    @FarmCoordinate = NULL,
                    @FarmLocation = NULL,
                    @FarmArea = NULL,
                    @FarmCreationDate = NULL,
                    @MasterLoginId = NULL,
                    @IsActive = NULL,
                    @EnterById = :master_login_id,
                    @ActionMode = 51,
                    @SearchCondition = NULL;
            """)

            result = connection.execute(query, {"master_login_id": master_login_id})
            rows = result.fetchall()

            if not rows:
                print("No Farm data found")
                return None

            columns = result.keys()
            farm_data_list = [dict(zip(columns, row)) for row in rows]

            return farm_data_list

    except Exception as e:
        print(f"Error executing SP_AMMasterFarm: {e}")
        return None

async def get_farm_details(id, farm_id):
    # Convert farm_id from string to integer
    farm_id = int(farm_id)

    engine = create_db_engine()
    try:
        with engine.connect() as connection:
            query = text("""
                SELECT * FROM AMMasterFarm 
                WHERE MasterFarmId = :farm_id AND MasterLoginId = :id
            """)
            result = connection.execute(query, {"id": id, "farm_id": farm_id})
            farm_data = result.fetchall()

            if not farm_data:
                print("No Farm data found")
                return None

            # Convert to list of dicts
            columns = result.keys()
            farm_data_dicts = [dict(zip(columns, row)) for row in farm_data]

            return farm_data_dicts
    except Exception as e:
        print(f"Error retrieving farm details: {e}")
        return None