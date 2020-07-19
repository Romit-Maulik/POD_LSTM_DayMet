# POD_LSTM_DayMet
POD-based daytime max temperature emulator for NASA DayMet data

# Important
1. I have not included numpy data - that stuff is too large to host here. I can transfer this to anyone interested. 
2. Respect the directory structure - I'm lazy and I haven't built in code to check and make directories if they aren't present.
3. The data set is large and training the convolutional autoencoder requires Horovod (which you can disable if you only want to use the POD-LSTM)

# Sample contour example
![Test snapshot](https://github.com/Romit-Maulik/POD_LSTM_DayMet/blob/master/Visualization/POD/Contours/Plot_test_0005.png "A test forecast")

# Sample POD coefficient plot for forecast
![Test POD Coefficients](https://github.com/Romit-Maulik/POD_LSTM_DayMet/blob/master/Visualization/POD/Coefficients/Coefficients_test.png "Test forecast of POD coefficients")


# Sample histogram and bias analyses (seven-day averages)
![Histograms - Central North America](https://github.com/Romit-Maulik/POD_LSTM_DayMet/blob/master/Analyses/POD/pdfs/WetSouth_116.png "Histograms on day 180 of 2016 (test)")

![Average forecasts - Central North America](https://github.com/Romit-Maulik/POD_LSTM_DayMet/blob/master/Analyses/POD/biases/Forecasts_CentralTundra.png "Test forecast of Central Tundra NA")
