# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 07:47:24 2019

@author: thompja
"""

import pandas as pd
import datetime
import math
import numpy as np

import xml.etree.ElementTree as ET

class tcx(object):
    
    def __init__(self):
        pass
        

    def distance(self, origin, destination):
        """
        From: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude/43211266#43211266
        Calculate the Haversine distance.
    
        Parameters
        ----------
        origin : tuple of float
            (lat, long)
        destination : tuple of float
            (lat, long)
    
        Returns
        -------
        distance_in_km : float
    
        Examples
        --------
        >>> origin = (48.1372, 11.5756)  # Munich
        >>> destination = (52.5186, 13.4083)  # Berlin
        >>> round(distance(origin, destination), 1)
        504.2
        """
        lat1, lon1 = origin
        lat2, lon2 = destination
        radius = 6371  # km
    
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) * math.sin(dlon / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = radius * c
    
        return d


    def parse_tcx(self, file_name, 
                  previous_trackpoint = {'cum_dur' : 0,
                                         'cum_dis' : 0,
                                         'prev_tim' : datetime.datetime.strptime('1970-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'),
                                         'prev_lat' : 0.0,
                                         'prev_lon' : 0.0,
                                         'prev_alt' : 0.0,
                                         'prev_gcum_dis' : 0.0}):  
                  
        """
        Extract time, lat, long, altitude and heartrate at each trackpoint.
        From this calculate speed and altitude increase by looking at the previous trackpoint.
        As the first trackpoint cannot calculate speed etc don't include in data output
        
        Also if an activity has been split over muliple files then want to compare the last trackpoint
        of the first file with the first trackpoint of the second file.
        
        Also need to deal with cases where the GPS has not kicked in and therefore no lat lon.
        """
        
        # if first point of file_name then need to treat differently
        first_point = True
        
        tcx_file = ET.parse(file_name)    
        
        # initialise in particular set the previous trackpoint which defaults to zero if first one 
        cum_dur = previous_trackpoint['cum_dur']
        cum_dis = previous_trackpoint['cum_dis']
        prev_tim = previous_trackpoint['prev_tim']
        prev_lat = previous_trackpoint['prev_lat']
        prev_lon = previous_trackpoint['prev_lon']
        prev_alt = previous_trackpoint['prev_alt']
        prev_gcum_dis = previous_trackpoint['prev_gcum_dis']
            
        root = tcx_file.getroot()
    
        # For the computers, the actual tag names are really long
        # We'll just use `tcx:` and the short name instead.
        ns = {
            'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
            'ns3' : 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'
        }
    
        trackpoints = root.findall('./tcx:Activities/tcx:Activity/tcx:Lap/tcx:Track/tcx:Trackpoint', ns)
    
        data = []
        for trackpoint in trackpoints:
            data_issue = False
            
            try:
                tim = trackpoint.find('./tcx:Time', ns).text
                tim = tim[:10] + ' ' + tim[11:-1]
                tim = datetime.datetime.strptime(tim, '%Y-%m-%d %H:%M:%S.%f')
            except:
                data_issue = True
            
            try:
                lat = float(trackpoint.find('./tcx:Position/tcx:LatitudeDegrees', ns).text)
            except:
                data_issue = True
            
            try:
                lon = float(trackpoint.find('./tcx:Position/tcx:LongitudeDegrees', ns).text)
            except:
                data_issue = True
            
            try:
                alt = float(trackpoint.find('./tcx:AltitudeMeters', ns).text)
            except:
                data_issue = True
            
            try:
                bpm = int(trackpoint.find('./tcx:HeartRateBpm/tcx:Value', ns).text)
            except:
                data_issue = True
                
            try:
                gcum_dis = float(trackpoint.find('./tcx:DistanceMeters', ns).text)
            except:
                gcum_dis = -99.0

            try:                
                gspeed = float(trackpoint.findall('./tcx:Extensions/ns3:TPX/ns3:Speed', 
                                                  ns)[0].text)
            except:
                data_issue = True
            
            # ignore trackpoints where any of the attributes are missing
            if data_issue:
                pass
            else:        
                # looks like the start of a new track has the same data as the end of the prvious track
                # so ignore this data as causes div zero issues
                if tim == prev_tim:
                    # data is same as previous trackpoint so ignore
                    pass
                else:
                    if not first_point:
                        # not the first trackpoint
                        dis = self.distance((prev_lat, prev_lon), (lat, lon))
                        cum_dis += dis
                        prev_lat = lat
                        prev_lon = lon
    
                        dur = (tim - prev_tim).total_seconds()
                        cum_dur += dur
                        prev_tim = tim
    
                        alt_diff = alt - prev_alt
                        prev_alt = alt
    
                        speed = 3600 * dis / dur  
                        if dis == 0:
                            alt_pc = 0
                        else:
                            alt_pc = 100 * (alt_diff / (dis * 1000))
                        
                        # Garmin distance in meters
                        # convert to km
                        gcum_dis /= 1000
                        
                        gdis = gcum_dis - prev_gcum_dis
                        prev_gcum_dis = gcum_dis                        
                        
                        
                        # Garmin speed measured in meters per second
                        # convert to km per hr
                        gspeed *= (3600 / 1000)                         
                        
                        data.append([dis, gdis, cum_dis, gcum_dis, 
                                     tim, dur, cum_dur, 
                                     lat, lon, alt, alt_diff, alt_pc, bpm, 
                                     speed, gspeed])  
                    else:
                        # first trackpoint 
                        first_point = False                        
                        prev_tim = tim
                        prev_lat = lat
                        prev_lon = lon
                        prev_alt = alt
                        prev_gcum_dis = gcum_dis
                            
        data_df = pd.DataFrame(data, columns=['dis', 'gdis', 'cum_dis', 'gcum_dis', 
                                              'tim', 'dur', 
                                              'cum_dur', 'lat', 'lon', 'alt', 
                                              'alt_diff', 'alt_pc', 'bpm', 
                                              'speed', 'gspeed'])
        
        last_trackpoint = {'cum_dur' : cum_dur,
                           'cum_dis' : cum_dis,
                           'prev_tim' : tim,
                           'prev_lat' : lat,
                           'prev_lon' : lon,
                           'prev_alt' : alt,
                           'prev_gcum_dis' : gcum_dis}
    
        return data_df, last_trackpoint    
    
    
    def gradient_summary(self, data_df, zero_speed_threshold, data_label, 
                         gradient_label, min_grad, max_grad, output=True,
                         pa_stop_speed=25, garmin_speed=True):
        """
        Summarise the data for a specific gradient range (measured as a ratio)
        If garmin_speed=True then use the Garmin speed at each point to decide
        if point is zero etc otherwise use calculated.
        
        data_df: has the data
        zero_speed_threshold: anything below this speed (kph) reported seperately
        data_label: added as a postfix to column headings in output dataframe
        gradient_label: used as row headings in output dataframe
        min_grad, max_grad: range of gradients to summarise
        output: printed summary required?        
        """
        
        if garmin_speed:
            speed_col_name = 'gspeed'
        else:
            speed_col_name = 'speed'
    
        
        totals = data_df.sum()
        zero_totals = data_df[data_df[speed_col_name] < zero_speed_threshold].sum()
        nonzero_totals = data_df[data_df[speed_col_name] >= zero_speed_threshold].sum()
        
        
        x_totals = data_df[(data_df['alt_pc'] <= max_grad) &
                            (data_df['alt_pc'] > min_grad)].sum()
    
        zero_totals = data_df[(data_df['alt_pc'] <= max_grad) &
                                (data_df['alt_pc'] > min_grad) &
                                (data_df[speed_col_name] < zero_speed_threshold)].sum()
        
        no_assist_totals = data_df[(data_df['alt_pc'] <= max_grad) &
                        (data_df['alt_pc'] > min_grad) &
                        (data_df[speed_col_name] > pa_stop_speed)].sum()
    
        nonzero_totals = data_df[(data_df['alt_pc'] <= max_grad) &
                                   (data_df['alt_pc'] > min_grad) & 
                                   (data_df[speed_col_name] >= zero_speed_threshold)].sum()
        
        nonzero_means = data_df[(data_df['alt_pc'] <= max_grad) &
                                (data_df['alt_pc'] > min_grad) & 
                                (data_df[speed_col_name] >= zero_speed_threshold)].mean()
        
        if nonzero_totals['dur'] > 0:
            speed = nonzero_totals['dis'] / (nonzero_totals['dur'] / 3600.0)
        else:
            speed = 0
    
        bpm_data_df = data_df[(data_df['alt_pc'] <= max_grad) &
                                (data_df['alt_pc'] > min_grad) & 
                                (data_df[speed_col_name] >= zero_speed_threshold)].loc[:, ['dur', 'bpm']]
        if bpm_data_df.shape[0] > 0:
            avg_bpm = np.average(bpm_data_df['bpm'], weights=bpm_data_df['dur'])
        else:
            avg_bpm = 0
        
        if totals['dur'] > 0:
            dur_pc = nonzero_totals['dur'] / totals['dur'] 
        else:
            dur_pc = 0
        
        if totals['dis'] > 0:
            dis_pc = nonzero_totals['dis'] / totals['dis'] 
        else:
            dis_pc = 0        
        
        if x_totals['dur'] > 0:
            zero_speed_pc = zero_totals['dur'] / x_totals['dur']
            non_assist_pc = no_assist_totals['dur'] / x_totals['dur']
            assist_pc = 1 - (zero_speed_pc + non_assist_pc)
        else:
            zero_speed_pc = 0
            non_assist_pc = 0
            assist_pc = 0
            
        if x_totals['dis'] > 0:
            zero_speed_dis_pc = zero_totals['dis'] / x_totals['dis']
            non_assist_dis_pc = no_assist_totals['dis'] / x_totals['dis']
            assist_dis_pc = 1 - (zero_speed_dis_pc + non_assist_dis_pc)
        else:
            zero_speed_dis_pc = 0
            non_assist_dis_pc = 0
            assist_dis_pc = 0
        
        return_df = pd.DataFrame([[speed, avg_bpm, dur_pc, dis_pc,
                                   zero_speed_pc, assist_pc, non_assist_pc,
                                   zero_speed_dis_pc, assist_dis_pc, non_assist_dis_pc]], 
                                 index=[gradient_label],
                                 columns=['speed_' + data_label, 
                                          'bpm_' + data_label,
                                          'dur_' + data_label,
                                          'dis_' + data_label,
                                          'zeropc_' + data_label,
                                          'assistpc_' + data_label,
                                          'nonpc_' + data_label,
                                          'dis_zeropc_' + data_label,
                                          'dis_assistpc_' + data_label,
                                          'dis_nonpc_' + data_label])
                
        if output:
        
            print()        
            print(data_label + ' : ' + gradient_label)
            print('--------------------------------')
            print('Total ascent (m): {:,.1f}'.format(x_totals['alt_diff']))
            print('Total distance (km, %): {:,.1f}, {:.1%}'.format(x_totals['dis'], x_totals['dis'] / totals['dis']))
            print('Total duration (hours, %): {:,.1f}, {:.1%}'.format(x_totals['dur'] / 3600.0, x_totals['dur'] / totals['dur']))
            print('Avg Speed (km/hr): {:,.1f}'.format(x_totals['dis'] / (x_totals['dur'] / 3600.0)))
            print('Zero Speed: Distance (km, %): {:,.1f}, {:.1%}'.format(zero_totals['dis'], 
                                                                            zero_totals['dis'] / totals['dis']))
            print('Zero Speed: Duration (hours, %): {:,.1f}, {:.1%}'.format(zero_totals['dur'] / 3600.0, 
                                                                            zero_totals['dur'] / totals['dur']))
            print('Non Zero Speed: Avg BPM: {:,.1f}'.format(nonzero_means['bpm']))
            print('Non Zero Speed: Distance (km, %): {:,.1f}, {:.1%}'.format(nonzero_totals['dis'], 
                                                                             nonzero_totals['dis'] / totals['dis']))
            print('Non Zero Speed: Duration (hours, %): {:,.1f}, {:.1%}'.format(nonzero_totals['dur'] / 3600.0, 
                                                                                nonzero_totals['dur'] / totals['dur']))
            print('Non Zero Speed: Avg Speed (km/hr): {:,.1f}'.format(nonzero_totals['dis'] / 
                                                                      (nonzero_totals['dur'] / 3600.0)))
            
        return return_df
    
    
    def overall_summary(self, data_df, zero_speed_threshold, pa_stop_speed, 
                        flat_gradient=0.02):
        """
        Summarise the data in data_df:
            avg speed (km per hr)
            total distance (km)
            total time (km)
            total ascent (m)
            total descent (m)
            total and percent of time less than zero_speed_threshold
            total and percent of time greater than pa_stop_speed
            
            flat avg speed (flat_gradient number)
            percent distance and duration flat
        """
    
        tot_dis = data_df['dis'].sum()
        tot_dur = data_df['dur'].sum()
        avg_speed = 3600 * tot_dis / tot_dur
        
        tot_asc = data_df[data_df['alt_diff'] > 0]['alt_diff'].sum()
        tot_desc = data_df[data_df['alt_diff'] < 0]['alt_diff'].sum()
        
        zero_dur_pc = (data_df[data_df['speed'] < zero_speed_threshold]['dur'].sum() /
                       tot_dur)
        zero_dis_pc = (data_df[data_df['speed'] < zero_speed_threshold]['dis'].sum() /
                       tot_dis)                    
        
        nonpa_dur_pc = (data_df[data_df['speed'] > pa_stop_speed]['dur'].sum() /
                       tot_dur)
        nonpa_dis_pc = (data_df[data_df['speed'] > pa_stop_speed]['dis'].sum() /
                       tot_dis)

        pa_dur_pc = 1 - (zero_dur_pc + nonpa_dur_pc)
        pa_dis_pc = 1 - (zero_dis_pc + nonpa_dis_pc)  

        flat_dis = data_df[(data_df['alt_pc'] > -flat_gradient) &
                           (data_df['alt_pc'] < flat_gradient)]['dis'].sum()
        
        flat_dur = data_df[(data_df['alt_pc'] > -flat_gradient) &
                           (data_df['alt_pc'] < flat_gradient)]['dur'].sum()
        
        flat_avg_speed = 3600 * flat_dis / flat_dur
        flat_dis_pc = flat_dis / tot_dis
        flat_dur_pc = flat_dur / tot_dur

        
        
        return_dic = {'tot_dis' : tot_dis,
                      'tot_dur' : tot_dur,
                      'avg_speed' : avg_speed,
                      'tot_asc' : tot_asc,
                      'tot_desc' : tot_desc,
                      'zero_dur_pc' : zero_dur_pc,
                      'zero_dis_pc' : zero_dis_pc,
                      'pa_dur_pc' : pa_dur_pc,
                      'pa_dis_pc' : pa_dis_pc,
                      'nonpa_dur_pc' : nonpa_dur_pc,
                      'nonpa_dis_pc' : nonpa_dis_pc,
                      'flat_avg_speed' : flat_avg_speed,
                      'flat_dis_pc' : flat_dis_pc,
                      'flat_dur_pc' : flat_dur_pc}
        
        return return_dic



    def lap_summary(self, data_df, interval, zero_speed_threshold, 
                    pa_stop_speed, flat_gradient=0.02):
        """
        Produce summary stats for each lap where a lap is based upon
        a specified interval.
        
        The interval laps can either start at the start of the data and go
        forwards or start at the end and go backwards. 
        
        Summary Metrics:
            those in overall_summary
        
        """    
                
        from_dist = 0
        end_of_data = False
        lap_summary = {}
        
        while not end_of_data:
            to_dist = from_dist + interval               
            
            df = data_df[(data_df['cum_dis'] >= from_dist) &
                         (data_df['cum_dis'] < to_dist)]
            
            if df.shape[0] == 0:
                end_of_data = True
            else:
                summary = self.overall_summary(df, zero_speed_threshold, 
                                               pa_stop_speed, flat_gradient) 
                
                lap_summary[to_dist] = summary
                from_dist = to_dist
        
        return lap_summary
            
            