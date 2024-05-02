# versuch1
Shasha li Robort project exercise https://git.uni-due.de/soankusu/cps-notebook/-/tree/main/Versuch1

## 1.3 create a package

### Problem 
```shell
sosaliii@lab11:~$ roscd
sosaliii@lab11:/opt/ros/noetic$ cd ..
sosaliii@lab11:/opt/ros$ catkin_create_pkg my_package rospy
Traceback (most recent call last):
  File "/usr/bin/catkin_create_pkg", line 33, in <module>
    sys.exit(load_entry_point('catkin-pkg==1.0.0', 'console_scripts', 'catkin_create_pkg')())
  File "/usr/lib/python3/dist-packages/catkin_pkg/cli/create_pkg.py", line 63, in main
    create_package_files(target_path=target_path,
  File "/usr/lib/python3/dist-packages/catkin_pkg/package_templates.py", line 216, in create_package_files
    _safe_write_files(newfiles, target_path)
  File "/usr/lib/python3/dist-packages/catkin_pkg/package_templates.py", line 189, in _safe_write_files
    os.makedirs(dirname)
  File "/usr/lib/python3.8/os.py", line 223, in makedirs
    mkdir(name, mode)
PermissionError: [Errno 13] Permission denied: '/opt/ros/my_package'
```

### Solution
```shell
vi ~/.bashrc
```
```shell
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:~/[Projekt folder]/
```
```shell
echo $ROS_PACKAGE_PATH
```
