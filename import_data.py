def _import_data():

	import os
	ls = os.listdir()
	dir_ls =[]

	for i in ls: dir_ls.append(name)

	database=[]
        	
	if len(dir_ls)==1:    
            data_file = str(dir_ls)
	    try:
	        if data_file[1][-3:]==str('csv'):
            	    data_p_values = pd.read_csv(file,delimiter=',', names=['Channel', 'Intensity'], skiprows = 21)
       	            de = 20.1*(10**-3)
        	    channel_p = data_p_values.Channel*de
              	    energy_p = (channel_p[channel_p<=24.0])
            	    intensity_p = (data_p_values.Intensity[channel_p<=24.0])
            	    database.append([energy_p, intensity_p])
            	
		    return database
	        else:
		    raise NameError('Document type {.csv, .xlsx. .dat, .txt}')
	     except NameError:
	        print('Not in a valid file type')
		raise
        else:
            for file in dir_ls:
                    data_p_values = pd.read_csv(file,delimiter=',', names=['Channel', 'Intensity'], skiprows = 21)
                    de = 19.8*(10**-3)
                    energy_p = data_p_values.Channel*de
                    intensity_p = data_p_values.Intensity
                    database.append([energy_p, intensity_p])
            return database
