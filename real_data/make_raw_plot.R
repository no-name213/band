# rm(list = ls())
library(here)
library(scales)   # for pretty_breaks()
source(here("real_data/data_process.R"))

# data_processed is loaded from here("real_data/data_process.R")
data <- as.matrix(data_processed)

# Data diagnosis
DIM(data) # 799 9
rownames(data) # corresponding dates
colnames(data) # fred-md codes


# ---------------------------------------------------------
# Demonstrate nonstationarity
# ---------------------------------------------------------
SAVE <- TRUE

coordinate_set <- c("FEDFUNDS", "UNRATE")

file_name <- paste(paste(coordinate_set, collapse = "_"),
                   "_nonstationary",
                   ".png", sep = "")
file_path <- here::here("real_data", file_name)

if (SAVE) {
  # Open PNG device with wider aspect ratio
  png(file_path, width = 1800, height = 800, res = 150)  # width > height
}

period = c(as.Date("1/1/2010", format = "%m/%d/%Y"),
           as.Date("8/1/2025", format = "%m/%d/%Y"))

# Extract series and dates
y <- data[, coordinate_set]
dates <- as.Date(rownames(data), format = "%m/%d/%Y")

par(mar = c(5, 8, 4, 2))   # increase left margin for large ylab text


cols <- rgb(0, 0, 0, alpha = 1)
plot(
  y,
  type = "p",
  pch = 16,
  cex = 1.2,
  cex.lab = 2.1,    # enlarge xlab & ylab
  cex.axis = 1.9,   # enlarge tick labels
  xlab = coordinate_set[1],
  ylab = coordinate_set[2],
  # main = paste("Time Series of", coordinate_set),
  col = cols,
  lwd = 1.5,
)

# Find indices inside the shading period
highlight_idx <- which(dates >= period[1] & dates <= period[2])

# Overlay highlighted points
highlight_col <- rgb(0.8, 0.3, 0, alpha = 1)  # semi-transparent brown

points(
  y[highlight_idx, 1],
  y[highlight_idx, 2],
  pch = 16,
  cex = 1.2,
  col = highlight_col
)
dev.off()


# Output the meta data
cat(
  "Black dots are from", 
  paste(format(as.Date(range(rownames(data)), format = "%m/%d/%Y"), "%Y-%m-%d"), collapse = " to "),
  "\nRed dots are from", 
  paste(format(period, "%Y-%m-%d"), collapse = " to "),
  "\n"
)

###
###
###


# ---------------------------------------------------------
# RAW FIGURES FOR THREE UNIVARIATE VARIABLES
# ---------------------------------------------------------
for (coordinate_set in c("CPI_RATE", "FEDFUNDS_CHANGE", "UNRATE_CHANGE")) {
  
  if (coordinate_set == "CPI_RATE") {
    main_name <- expression(IR[t])
  } else if (coordinate_set == "FEDFUNDS_CHANGE") {
    main_name <- expression(FR[t])
  } else {
    main_name <- expression(UR[t])
  }
  
  
  
    
  
  
  file_name <- paste(paste(coordinate_set, collapse = "_"),
                     "_raw_data",
                     ".png", sep = "")
  file_path <- here::here("real_data", file_name)
  
  if (SAVE) {
    # Open PNG device with wider aspect ratio
    png(file_path, width = 1800, height = 800, res = 150)  # width > height
  }
  
  
  
  period1 = c(as.Date("1/1/2014", format = "%m/%d/%Y"),
              as.Date("12/1/2019", format = "%m/%d/%Y"))
  period2 = c(as.Date("8/1/2022", format = "%m/%d/%Y"),
              as.Date("8/1/2025", format = "%m/%d/%Y"))
  # To make figure beautiful, the tuing periods are extended by one month:
  period3 = c(as.Date("1/1/2010", format = "%m/%d/%Y"),
              as.Date("2/1/2014", format = "%m/%d/%Y"))
  period4 = c(as.Date("6/1/2021", format = "%m/%d/%Y"),
              as.Date("8/1/2022", format = "%m/%d/%Y"))
  
  
  # Extract series and dates
  y <- data[, coordinate_set]
  dates <- as.Date(rownames(data), format = "%m/%d/%Y")
  
  # ---------------------------------------------------------
  # Base plot: time series with readable date axis
  # ---------------------------------------------------------
  plot(
    dates, y,
    type = "l",
    xlab = "",
    ylab ="",
    # main = paste("Time Series of", coordinate_set),
    col = "black",
    lwd = 1.5,
    xaxt = "n",
    main = main_name,
    cex.lab = 2.1,    # enlarge xlab & ylab
    cex.axis = 2.4,   # enlarge tick labels
    cex.main = 2.5,   # enlarge main title
    tck = 0.1,  # tick length (small positive number → inside)
    mgp = c(3, 1.5, 0) # axis title distance, tick label distance
  )
  
  # Clean date axis (1 label per ~2 years)
  axis(1, 
       at = pretty(dates, n = 10), 
       labels = format(pretty(dates, n = 10), "%Y"), 
       cex.axis = 2.5,
       tck = 0.1)
  
  # ---------------------------------------------------------
  # Add transparent points for the selected period
  # ---------------------------------------------------------
  # find indices that fall into the period
  # Clean x-axis
  
  # ---------------------------------------------------------
  # Add markers for START of each period on the x-axis
  # ---------------------------------------------------------
  # Helper function to shade a date range
  shade_period <- function(start_date, end_date, col = "gray90") {
    usr <- par("usr")
    rect(start_date, usr[3], end_date, usr[4],
         col = col, border = NA)
  }
  
  # Shade period1 (light brown)
  shade_period(period1[1], period1[2], col = rgb(0.7, 0.5, 0.2, 0.6))
  
  # Shade period1 (light brown)
  shade_period(period3[1], period3[2], col = rgb(0.7, 0.5, 0.2, 0.3))
  
  # Shade period2 (teal)
  shade_period(period2[1], period2[2], col = rgb(0, 0.6, 0.5, 0.6))
  
  # Shade period1 (light brown)
  shade_period(period4[1], period4[2], col = rgb(0, 0.6, 0.5,  0.3))
  
  # Re-draw the line on top (so shading stays behind)
  lines(dates, y, col = "black", lwd = 1.3)
  
  dev.off()
  
  
  # Output the meta data
  cat(
    "Brown shaded area ranging from",
    paste(format(period1, "%Y-%m-%d"), collapse = " to "),
    "\nGreen shaded area ranging from",
    paste(format(period2, "%Y-%m-%d"), collapse = " to "),
    "\n"
  )
}
