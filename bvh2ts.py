from imusim.all import *
import imusim
import numpy as np
from tqdm import tqdm
import multiprocessing
import os

with open('./bvh/000000.bvh', 'r') as file:
    lines = file.readlines()
    line_109 = lines[108]
    frame_time = line_109.split(': ')[1].strip() 
    frame_time_value = float(frame_time)
    print(frame_time_value)

def process_file(f):

    imu_file_path = './output/%s.npy' % f
    if not os.path.exists(imu_file_path):
        
        samplingPeriod = frame_time_value
        imu = Orient3IMU()
        env = Environment()

        samples = 1000
        rotationalVelocity = 20
        calibrator = ScaleAndOffsetCalibrator(env, samples, samplingPeriod, rotationalVelocity)
        calibration = calibrator.calibrate(imu)

        try:
            model = loadBVHFile('./bvh/%s.bvh' % f)
            splinedModel = SplinedBodyModel(model)

            imu_list = []
            for i in range(22):
                sim = Simulation(environment=env)
                imu.simulation = sim
            
                if i not in [4,8,13,17,21]:
                    imu.trajectory = splinedModel.getJoint('joint_%s' % str(i))
                else:
                    imu.trajectory = splinedModel.getPoint('joint_%s_end' % str(i-1))

                sim.time = splinedModel.startTime
                BasicIMUBehaviour(imu, samplingPeriod, calibration, initialTime=sim.time)
                sim.run(splinedModel.endTime, printProgress=False)

                acc = imu.accelerometer.calibratedMeasurements.values
                gyro = imu.gyroscope.calibratedMeasurements.values

                imu_npy = np.concatenate((acc, gyro), axis=0)
                imu_list.append(imu_npy)

            imu_npy = np.stack(imu_list, axis=1).transpose(2,1,0)
            np.save('./output/%s' % f, imu_npy)

        except (imusim.maths.splines.Spline.InsufficientPointsError, AttributeError, IndexError) as e:
            print(f"Error processing file {f}: {e}. Skipping.")
            with open('log.txt', 'a') as log_file:
                log_file.write(f + '\n')

source_dir = './bvh'
npy_files = [file[:-4] for file in os.listdir(source_dir) if file.endswith('.bvh')]

# Process files in parallel
pool = multiprocessing.Pool(processes=8)
for _ in tqdm(pool.imap_unordered(process_file, npy_files), total=len(npy_files)):
    pass
pool.close()
pool.join()