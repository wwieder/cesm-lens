import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_ffmc(tmeanC, wind_kmhr, RH, precip_mm, ffmc_prev):
        #  Calculates the Canadian Forest Fire Rating System (CFRS) 
        #  fine fuel moisture code (ffmc)
        #  Adapted from Wang 2015 NRC Information Report NOR-X-424
        #
        #  Record of revisions:
        #      Date       Programmer          Description of change
        #      ====       ==========          =====================
        #    05/15/20     A. C. Foster         Original Code
        #    02/09/20     J.K. Shuman          Updated for python

    # fine fuel moisture content from previous day
    Mo = (147.2 * (101.0 - ffmc_prev)) / (59.5 + ffmc_prev)
    
    if (RH > 100.0): RH = 100.0

    if (precip_mm > 0.5):
        
        #account for loss in canopy
        rainfall = precip_mm - 0.5
        if (Mo > 150.0):
            Mo = (Mo + 42.5 * rainfall * np.exp(-100.0 / (251.0 - Mo)) * (1.0 - np.exp(-6.93 / rainfall))) \
            + (0.0015 * (Mo - 150.0) ** 2.0) * (rainfall ** 0.5)
        else:   #Mo<= 150.0
            Mo = Mo + 42.5 * rainfall * np.exp(-100.0 /(251.0 - Mo)) * (1.0 - np.exp(-6.93 /rainfall))
                   
    if (Mo > 250.0):
        Mo = 250.0

    # fine fuel equilibrium moisture content (EMC) for drying
    edry = 0.942 * (RH ** 0.679) + (11.0 * np.exp((RH - 100.0) / 10.0)) + 0.18 * (21.1 - tmeanC) * \
                    (1.0 - 1.0/np.exp(0.1150 * RH))            

    # fine fuel equilibrium moisture content from wetting
    ewet = 0.618 * (RH ** 0.753) + (10.0 * np.exp((RH - 100.0) / 10.0)) + 0.18 * (21.1 - tmeanC) * \
                    (1.0 - 1.0/np.exp(0.115 * RH))

    # log drying rate at normal temperature (21.1 degC)
    if (Mo < edry) and (Mo < ewet):
        k1 = 0.424 * (1.0 - (((100.0 - RH)/100.0) ** 1.7)) + 0.0694 * (wind_kmhr ** 0.5) * \
                (1.0 - ((100.0 - RH) / 100.0) ** 8.0)         
    else:
        k1 = 0.0

    # effect of temperature on drying ratio
    kw = k1 * (0.581 * np.exp(0.0365 * tmeanC))                       

    # calculate moisture after drying
    if (Mo < edry) and (Mo < ewet):
        mk = ewet - (ewet - Mo)/(10.0**kw)                      
    else:
        mk = Mo

    # log of wetting rate at normal temperature (21.1 degC)
    if (Mo > edry):
        k1 = 0.424 * (1.0 - (RH /100.0) ** 1.7) + 0.0694 * (wind_kmhr ** 0.5) * \
                (1.0 - (RH / 100.0) ** 8.0)

    # effect of temperature on wetting rate
    kw = k1 * (0.581 * np.exp(0.0365 * tmeanC))

    # calculate moisture after wetting
    if (Mo > edry):
        mk = edry + (Mo -edry)/(10.0**kw)                      

    # calculate ffmc and correct for outside of bounds
    ffmc = (59.5 * (250.0 - mk)) / (147.2 + mk)                    
    
    if ffmc > 101.0:
        ffmc = 101.0
    if ffmc <= 0.0:
        ffmc = 0.0
        
    return ffmc


def calc_dmc(tmaxC, RH, precip_mm, dmc_prev, month, latitude):
        #  Calculates the Canadian Forest Fire Rating System (CFRS) 
        #  duff moisture code (dmc)
        #  Adapted from Wang 2015 NRC Information Report NOR-X-424
        #
        #  Record of revisions:
        #      Date       Programmer          Description of change
        #      ====       ==========          =====================
        #    05/15/20     A. C. Foster         Original Code
        #    02/09/20     J.K. Shuman          Updated for python
        
    # Data dictionary: constants
    # Day length adjustments
    # For latitude near equation (-10, 10 degrees), use a factor of 9
    # for all months
    # day length by lat
    ell01 = np.array([6.5,7.5,9.0,12.8,13.9,13.9,12.4,10.9,9.4,8.0,7.0,6.0])   # lat >= 30N
    ell02 = np.array([7.9, 8.4, 8.9, 9.5, 9.9,10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8]) # 30 > latitude >= 10
    ell03 = np.array([10.1, 9.6, 9.1, 8.5, 8.1,7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2]) # -10 > latitude >= -30
    ell04 = np.array([11.5, 10.5, 9.2, 7.9, 6.8,6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8]) # latitude < -30
 
    if (RH > 100.0): RH = 100.0
    
    if (tmaxC < -1.1):
        temp0 = -1.1
    else:
        temp0 = tmaxC

    # Determine day length adjustment based on latitude
    if (latitude > 30.0):
        ell = ell01
    elif (latitude <= 30.0 and latitude > 10.0):
        ell = ell02
    elif (latitude <= 10.0 and latitude > -10.0):
        ell = 9.0
    elif (latitude <= -10.0 and latitude > -30.0):
        ell = ell03
    elif (latitude <= -30.0):
        ell = ell04

    # Log drying rate
    rk = 1.894*(temp0 + 1.1)*(100.0 -RH)*(ell[month-1]*0.0001)

    # Net rainfall (mm)
    rw = 0.92*precip_mm - 1.27

    # Alteration to EQ 12 to calculate more accurately
    wmi = 20.0 + 280.0/np.exp(0.023*dmc_prev)

    # EQ 13a-c
    if (dmc_prev <= 33.0):
        b = 100.0/(0.5 + 0.3*dmc_prev)
    elif (dmc_prev > 33.0) and (dmc_prev <= 65.0):
        b= 14.0 - 1.3*np.log(dmc_prev)
    else:
        b = 6.2*np.log(dmc_prev) - 17.2

    # Moisture content after rain
    wmr = wmi + (1000.0*rw) / (48.77+b*rw)

    # constrain precip
    if (precip_mm <= 1.5):
        pr = dmc_prev
    else:
        pr = 43.43*(5.6348 - np.log(wmr - 20.0))

    if (pr < 0.0): pr = 0.0

    # calculate dmc
    dmc = pr + rk
    
    if (dmc < 0.0): dmc = 0.0

    return dmc


def calc_dc(tmaxC, precip_mm, dc_prev, month, latitude):
        #  Calculates the Canadian Forest Fire Rating System (CFRS) 
        #  drought code (dc)
        #  Adapted from Wang 2015 NRC Information Report NOR-X-424
        #
        #  Record of revisions:
        #      Date       Programmer          Description of change
        #      ====       ==========          =====================
        #    05/15/20     A. C. Foster         Original Code
        #    02/09/20     J.K. Shuman          Updated for python
        
    # Data dictionary: constants
    # Day length adjustments
    # Near equator, just use 1.4 for all months
    fl01 = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])# lat > 20N
    fl02 = np.array([6.4, 5.0, 2.4, 0.4, -1.6,-1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8])# lat < -20
    
    precip = precip_mm #rainfall to mm
    
    if (tmaxC < -2.8):
        temp0 = -2.8
    else:
        temp0 = tmaxC
    
    # Potential evapotransipiration (mm) EQ 22
    if (latitude > 20.0):
        pe = (0.36*(temp0 +2.8) + fl01[-1])/2.0  
    elif (latitude <= -20.0):
        pe = (0.36*(temp0 +2.8) + fl02[-1])/2.0  
    else:
        pe = (0.36*(temp0 +2.8) + fl01[-1])/2.0  
        
    # cap pe at 0 for negative winter DC values
    if (pe < 0.0): pe = 0.0
       
    ra = precip
    
    # effective rainfall
    rw = 0.83*ra -1.27
    
    # EQ 19
    smi = 800.0 * np.exp(-1.0*dc_prev/400.0)
    
    # Modify EQ 21
    dr0 = dc_prev - 400.0*np.log( 1.0 +((3.937*rw)/smi)) # EQ 20 & 21
    if (dr0 < 0.0): dr0 = 0.0
    
    # drying rate, use yesterday's DC if precip < 2.8
    if (precip <= 2.8):
        dr = dc_prev
    else:
        dr = dr0
    
    # calculate dc
    dc = dr + pe
    
    if (dc < 0.0): dc = 0.0
        
    return dc


def calc_isi(wind_kmhr,ffmc):
        #  Calculates the Canadian Forest Fire Rating System (CFRS) 
        #  initial spread index (isi)
        #  Adapted from Wang 2015 NRC Information Report NOR-X-424
        #
        #  Record of revisions:
        #      Date       Programmer          Description of change
        #      ====       ==========          =====================
        #    02/09/20     J.K. Shuman         Original code   
    
    # moisture content fine fuels
    mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    # fine fuel moisture function
    ff = 19.115 * np.exp(mo * -0.1386) * (1.0 + (mo**5.31)/49300000.0)
    # initial spread index
    isi = ff * np.exp(0.05039 * wind_kmhr)
    
    return isi


def calc_bui(dmc,dc):
        #  Calculates the Canadian Forest Fire Rating System (CFRS) 
        #  build up index (bui)
        #  Adapted from Wang 2015 NRC Information Report NOR-X-424
        #
        #  Record of revisions:
        #      Date       Programmer          Description of change
        #      ====       ==========          =====================
        #    02/09/20     J.K. Shuman         Original code   
    
    if dmc <= 0.4 * dc:
        bui = (0.8 *dc *dmc) / (dmc + 0.4 *dc)                                 # EQ 27a
    else:
        bui = dmc - (1.0 -0.8 *dc/(dmc + 0.4*dc)) * (0.92 +(0.0114*dmc)**1.7)  # EQ 27b
    
    if bui < 0.0:
        bui = 0.0
    
    return bui
    
    
def calc_fwi(isi,bui): 
        #  Calculates the Canadian Forest Fire Rating System (CFRS) 
        #  build up index (bui)
        #  Adapted from Wang 2015 NRC Information Report NOR-X-424
        #
        #  Record of revisions:
        #      Date       Programmer          Description of change
        #      ====       ==========          =====================
        #    02/09/20     J.K. Shuman         Original code   
        
    if bui <= 80.0:
        bb = 0.1 * isi * (0.626*bui**0.809 + 2.0)                 #EQ 28a
    else:
        bb = 0.1*isi*(1000.0/(25. + 108.64/np.exp(0.023*bui))) # EQ 28b
    if(bb <= 1.0):
        fwi = bb                                                 # EQ 30b
    else:
        fwi = np.exp(2.72 * (0.434*np.log(bb))**0.647)       # EQ 30a
    
    return fwi


def main():
    

    # get weather data
    weatherDat = pd.read_csv('/home/jkshuman/python_tutorial_NCAR/data/RAWS_processed_CP.csv')
    tmeanC = weatherDat['temp_mean.C']
    tmaxC = weatherDat['temp_max.C']
    RH = weatherDat['RH_mean']
    precip_mm = weatherDat['precip.mm']
    wind_kmhr = weatherDat['wind.kmh']
    day = weatherDat['doy']
    year = weatherDat['year']
    date = weatherDat['date'].tolist()

    month = [int(dt.split('/')[0])-1 for dt in date]
    wind_ms = wind_kmhr*1000.0/60.0/60.0
    
    #include latitude
    lati = np.zeros(weatherDat.shape[0])
    lati_initial = 66.0
    lati[lati==0] = lati_initial
 
    ffmc_daily = np.zeros(weatherDat.shape[0])
    dmc_daily = np.zeros(weatherDat.shape[0])
    dc_daily = np.zeros(weatherDat.shape[0])
    isi_daily = np.zeros(weatherDat.shape[0])
    bui_daily = np.zeros(weatherDat.shape[0])
    fwi_daily = np.zeros(weatherDat.shape[0])
    nesterov = np.zeros(weatherDat.shape[0])
    
    ffmc_daily[0] = 80.0        ## inititial values for fine fuel moisture code
    dmc_daily[0] = 6.0          ## initial values for duff moisture code
    dc_daily[0] = 15.0          ## initial values for drought code
         
    for i in range(1, weatherDat.shape[0]-1):

        # calculate nesterov index
        #nesterov[i] = calc_nesterov(tmaxC[i], RH[i], precip_mm[i], nesterov[i - 1])

        # calculate fine fuel moisture code
        ffmc_daily[i] = calc_ffmc(tmeanC[i], wind_kmhr[i], RH[i], precip_mm[i], ffmc_daily[i - 1])

        # calculate duff moisture code
        dmc_daily[i] = calc_dmc(tmaxC[i], RH[i], precip_mm[i], dmc_daily[i-1], month[i], lati[i])
        
        # calculate duff moisture code
        dc_daily[i] = calc_dc(tmaxC[i], precip_mm[i], dc_daily[i-1], month[i], lati[i])
        
        # calculate initial spread index
        isi_daily[i] = calc_isi(wind_kmhr[i], ffmc_daily[i])
        
        # calculate initial spread index
        bui_daily[i] = calc_bui(dmc_daily[i], dc_daily[i])
        
        # calculate initial spread index
        fwi_daily[i] = calc_fwi(isi_daily[i], bui_daily[i])
    
        

        
    years = pd.unique(weatherDat['year'])
    years.sort()
    for i in range(0, 8):
        ind = i*364 + i
        plt.plot(day[ind:ind+364], ffmc_daily[ind:ind+364], label=years[i])
    plt.legend(title='Year')
    plt.xlabel('Day of Year')
    plt.ylabel('Fine Fuel Moisture Code')
    plt.title('FFMC for Caribou Creak 2012-2019')
    plt.show()
    
    years = pd.unique(weatherDat['year'])
    years.sort()
    for i in range(0, 8):
        ind = i*364 + i
        plt.plot(day[ind:ind+364], dmc_daily[ind:ind+364], label=years[i])
    plt.legend(title='Year')
    plt.xlabel('Day of Year')
    plt.ylabel('Duff Moisture Code')
    plt.title('DMC for Caribou Creak 2012-2019')
    plt.show()
    
    years = pd.unique(weatherDat['year'])
    years.sort()
    for i in range(0, 8):
        ind = i*364 + i
        plt.plot(day[ind:ind+364], dc_daily[ind:ind+364], label=years[i])
    plt.legend(title='Year')
    plt.xlabel('Day of Year')
    plt.ylabel('Drought Code')
    plt.title('DC for Caribou Creak 2012-2019')
    plt.show()
    
    years = pd.unique(weatherDat['year'])
    years.sort()
    for i in range(0, 8):
        ind = i*364 + i
        plt.plot(day[ind:ind+364], isi_daily[ind:ind+364], label=years[i])
    plt.legend(title='Year')
    plt.xlabel('Day of Year')
    plt.ylabel('Initial Spread Index')
    plt.title('ISI for Caribou Creak 2012-2019')
    plt.show()
    
    years = pd.unique(weatherDat['year'])
    years.sort()
    for i in range(0, 8):
        ind = i*364 + i
        plt.plot(day[ind:ind+364], bui_daily[ind:ind+364], label=years[i])
    plt.legend(title='Year')
    plt.xlabel('Day of Year')
    plt.ylabel('Build Up Index')
    plt.title('BUI for Caribou Creak 2012-2019')
    plt.show()

    years = pd.unique(weatherDat['year'])
    years.sort()
    for i in range(0, 8):
        ind = i*364 + i
        plt.plot(day[ind:ind+364], fwi_daily[ind:ind+364], label=years[i])
    plt.legend(title='Year')
    plt.xlabel('Day of Year')
    plt.ylabel('Fire Weather Index')
    plt.title('FWI for Caribou Creak 2012-2019')
    plt.show()



if __name__ == '__main__':
    main()


