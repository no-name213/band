# rm(list = ls())
# data downloaded from https://research.stlouisfed.org/econ/mccracken/fred-databases/
library(here)

# Five auxiliary functions
DIM <- function(x) {
  if( is.null( dim(x) ) )
    return( length(x) )
  dim(x)
}


# Function to get the current level, adjusted for length after differencing
get_level <- function(x) { x[-1] } 
# Define Transformation Functions
# Reversing the names to match typical usage and your column names
# Used for _RATE
percent_change <- function(x) { (x[-1] - x[-length(x)]) / x[-length(x)]}
# Used for _CHANGE
first_difference <- function(x) { x[-1] - x[-length(x)] }

create_ar_lag_matrix <- function(data, 
                                 coordinate_set, 
                                 lag_period = c(0, 1)) {
  
  # Ensure input is a matrix
  data_mat <- as.matrix(data[, coordinate_set, drop = FALSE])
  
  n <- nrow(data_mat)
  p <- ncol(data_mat)
  
  lagged_list <- list()
  
  for (lag in lag_period) {
    
    if (lag == 0) {
      # no lag, use original
      lagged_mat <- data_mat
      colnames(lagged_mat) <- paste0(colnames(data_mat), "_lag0")
      
    } else {
      # lag k: first k rows are NA
      lagged_mat <- rbind(
        matrix(NA, nrow = lag, ncol = p),
        data_mat[1:(n - lag), , drop = FALSE]
      )
      colnames(lagged_mat) <- paste0(colnames(data_mat), "_lag", lag)
    }
    
    lagged_list[[length(lagged_list) + 1]] <- lagged_mat
  }
  
  # Combine everything
  lag_matrix <- do.call(cbind, lagged_list)
  
  # Remove any rows containing NA
  lag_matrix <- lag_matrix[complete.cases(lag_matrix), , drop = FALSE]
  
  return(lag_matrix)
}





# LOAD THE CSV FILE:
file <- here("real_data/2025-10-MD.csv")

data <- read.csv(file)
# DIM(data_) # 802, 127
# the first row is the category index.

# date : data[, 1]
# "Transform" : data[1,1]
# categories of each column : data[1,-1]

# Extract the category information (first row, excluding the first column)
categories <- data[1, -1]

# Save the first column (date) and remove it from the data
date <- data[, 1]
date <- date[-1] # length 801
data_ <- data[, -1]

# Remove the category row from the data (keep only actual observations)
data_ <- data_[-1, ]
DIM(data_) # 801, 126




# A missing value in unrate 2025:09
start_ <- which(date == "1/1/1959")
end_ <- which(date == "8/1/2025")
date_ <- date[start_:end_]
data_ <- data_[start_:end_,]


# The FEDFUNDS series in FRED-MD is the effective federal funds rate, 
# already expressed as an annualized percentage. 
# We use it directly as the U.S. short-term interest rate.
interest.index <- which(colnames(data_) == 'FEDFUNDS') # Interest rate
cpi.index <- which(colnames(data_) == 'CPIAUCSL') # CPIAUCSL is the CPI for all items
unrate.index <- which(colnames(data_) == "UNRATE") # UNRATE is the unemployement rate


# Extract Original Columns (Assuming data_ columns are in order)
fedfunds <- data_[, interest.index]
cpi <- data_[, cpi.index]
unrate <- data_[, unrate.index]

# 3. Create the Transformed Data Columns
# Note: All transformed series will be one observation shorter than the original levels.

# --- CPI (Inflation) ---
CPI = get_level(cpi)
CPI_CHANGE = first_difference(cpi) # Simple change (Delta x)
CPI_RATE = percent_change(cpi)   # Percentage change (Growth rate)

# --- FEDFUNDS (Interest) ---
FEDFUNDS = get_level(fedfunds)
FEDFUNDS_CHANGE = first_difference(fedfunds)
FEDFUNDS_RATE = percent_change(fedfunds)

# --- UNRATE (Unemployment) ---
UNRATE = get_level(unrate)
UNRATE_CHANGE = first_difference(unrate)
UNRATE_RATE = percent_change(unrate)


# 4. Combine into the Final Data Frame
# Note: We use the lagged levels (get_level) to ensure all series have the same length.
data_transformed <- data.frame(
  CPI = CPI,
  CPI_CHANGE = CPI_CHANGE,
  CPI_RATE = CPI_RATE,
  FEDFUNDS = FEDFUNDS,
  FEDFUNDS_CHANGE = FEDFUNDS_CHANGE,
  FEDFUNDS_RATE = FEDFUNDS_RATE,
  UNRATE = UNRATE,
  UNRATE_CHANGE = UNRATE_CHANGE,
  UNRATE_RATE = UNRATE_RATE
)

# Assign Final Column Names
colnames(data_transformed) <- c("CPI", "CPI_CHANGE", "CPI_RATE", 
                                "FEDFUNDS", "FEDFUNDS_CHANGE", "FEDFUNDS_RATE", 
                                "UNRATE", "UNRATE_CHANGE", "UNRATE_RATE")

data_processed <- data_transformed
# Final data from 2/1/1959 to 8/1/2025
rownames(data_processed) <- date_[-1]


DIM(data_processed)
length(date)
data_processed
DIM(data_processed) # 799 9
rownames(data_processed) # corresponding dates
colnames(data_processed) # fred-md codes
