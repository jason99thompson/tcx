# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 07:47:24 2019

@author: thompja
"""

import pandas as pd
import datetime
import math

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
    
    
def gradient_summary(self, data_df, zero_speed_threshold, 
                     gradient_label, min_grad, max_grad, output=True):
    """
    Summarise the data by gradient   
    
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
    

    
    if output:
    
        print()
        print(gradient_label)
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
        
