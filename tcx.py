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
                                         'prev_alt' : 0.0}):  
                  
        """
        Extract time, lat, long, altitude and heartrate at each trackpoint.
        From this calculate speed and altitude increase by looking at the previous trackpoint.
        As the first trackpoint cannot calculate speed etc don't include in data output
        
        Also if an activity has been split over muliple files then want to compare the last trackpoint
        of the first file with the first trackpoint of the second file.
        
        Also need to deal with cases where the GPS has not kicked in and therefore no lat lon.
        """
        
        tcx_file = ET.parse(file_name)    
        
        # initialise in particular set the previous trackpoint which defaults to zero if first one 
        cum_dur = previous_trackpoint['cum_dur']
        cum_dis = previous_trackpoint['cum_dis']
        prev_tim = previous_trackpoint['prev_tim']
        prev_lat = previous_trackpoint['prev_lat']
        prev_lon = previous_trackpoint['prev_lon']
        prev_alt = previous_trackpoint['prev_alt']        
            
        root = tcx_file.getroot()
    
        # For the computers, the actual tag names are really long
        # We'll just use `tcx:` and the short name instead.
        ns = {
            'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
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
                    if prev_lat > 0:
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
                        alt_pc = 100 * (alt_diff / (dis * 1000))
    
                        data.append([dis, cum_dis, tim, dur, cum_dur, 
                                     lat, lon, alt, alt_diff, alt_pc, bpm, speed])  
                    else:
                        # first trackpoint 
                        prev_tim = tim
                        prev_lat = lat
                        prev_lon = lon
                        prev_alt = alt
    
        data_df = pd.DataFrame(data, columns=['dis', 'cum_dis', 'tim', 'dur', 'cum_dur', 
                                              'lat', 'lon', 'alt', 'alt_diff', 'alt_pc', 'bpm', 'speed'])
        
        last_trackpoint = {'cum_dur' : cum_dur,
                           'cum_dis' : cum_dis,
                           'prev_tim' : tim,
                           'prev_lat' : lat,
                           'prev_lon' : lon,
                           'prev_alt' : alt}
    
        return data_df, last_trackpoint    
    
    
    def gradient_summary(self, data_df, zero_speed_threshold, data_label, 
                         gradient_label, min_grad, max_grad, output=True):
        """
        Summarise the data for a specific gradient range (measured as a ratio)
        
        data_df: has the data
        zero_speed_threshold: anything below this speed (kph) reported seperately
        data_label: added as a postfix to column headings in output dataframe
        gradient_label: used as row headings in output dataframe
        min_grad, max_grad: range of gradients to summarise
        output: printed summary required?        
        """
    
        totals = data_df.sum()
        zero_totals = data_df[data_df['speed'] < zero_speed_threshold].sum()
        nonzero_totals = data_df[data_df['speed'] >= zero_speed_threshold].sum()
        
        
        x_totals = data_df[(data_df['alt_pc'] <= max_grad) &
                            (data_df['alt_pc'] > min_grad)].sum()
    
        zero_totals = data_df[(data_df['alt_pc'] <= max_grad) &
                                (data_df['alt_pc'] > min_grad) &
                                (data_df['speed'] < zero_speed_threshold)].sum()
    
        nonzero_totals = data_df[(data_df['alt_pc'] <= max_grad) &
                                   (data_df['alt_pc'] > min_grad) & 
                                   (data_df['speed'] >= zero_speed_threshold)].sum()
        
        nonzero_means = data_df[(data_df['alt_pc'] <= max_grad) &
                                (data_df['alt_pc'] > min_grad) & 
                                (data_df['speed'] >= zero_speed_threshold)].mean()
        
        if nonzero_totals['dur'] > 0:
            speed = nonzero_totals['dis'] / (nonzero_totals['dur'] / 3600.0)
        else:
            speed = 0
    
        bpm_data_df = data_df[(data_df['alt_pc'] <= max_grad) &
                                (data_df['alt_pc'] > min_grad) & 
                                (data_df['speed'] >= zero_speed_threshold)].loc[:, ['dur', 'bpm']]
        if bpm_data_df.shape[0] > 0:
            avg_bpm = np.average(bpm_data_df['bpm'], weights=bpm_data_df['dur'])
        else:
            avg_bpm = 0
        
        if totals['dur'] > 0:
            dur_pc = nonzero_totals['dur'] / totals['dur'] 
        else:
            dur_pc = 0
        
        return_df = pd.DataFrame([[speed, avg_bpm, dur_pc]], 
                                 index=[gradient_label],
                                 columns=['speed_' + data_label, 
                                          'bpm_' + data_label,
                                          'dur_' + data_label])
                
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
    
    
    
    def lap_summary(data_df, interval, zero_speed_threshold, 
                    speed_zones, flat_gradient=0.02
                    start=True):
        """
        Produce summary stats for each lap where a lap is based upon
        a specified interval.
        
        The interval laps can either start at the start of the data and go
        forwards or start at the end and go backwards. start=True means 
        intervals based upon the start.
        
        Summary Metrics:
            avg speed
            flat avg speed (flat_gradient number)
        
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
                #start_df = df.iloc[0,:]
                #end_df = df.iloc[-1,:]                
                #dis = end_df['cum_dis'] - start_df['cum_dis']
                #dur = end_df['cum_dur'] - start_df['cum_dur']
                dis = df['dis'].sum()
                dur = df['dur'].sum()                
                avg_speed = dis / (dur / 3600)
                
                flat_dis = df[(df['alt_pc'] <= flat_gradient) &
                              (df['alt_pc'] >= -flat_gradient)].loc['dis'].sum()
                flat_dur = df[(df['alt_pc'] <= flat_gradient) &
                              (df['alt_pc'] >= -flat_gradient)].loc['dur'].sum()
                flat_avg_speed = flat_dis / (flat_dur / 3600)
                
                from_dist = to_dist
                
                lap_summary[to_dist] = (avg_speed, flat_avg_speed)
        
        return lap_summary
            
            