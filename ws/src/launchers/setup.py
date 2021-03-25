from setuptools import setup

package_name = 'launchers'

setup(
    name=package_name,
    version='0.1.5',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        # TODO: add launch files here once ported to ROS 2 (probably use glob)
        # (
        #     'share/' + package_name,
        #     ['launch/something.launch.py']
        # ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jaroslav Klapálek',
    maintainer_email='klapajar@fel.cvut.cz',
    description='TODO',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'launch_components = launchers.launch_components:main',
            'launch_sensors = launchers.launch_sensors:main',
        ],
    },
)
