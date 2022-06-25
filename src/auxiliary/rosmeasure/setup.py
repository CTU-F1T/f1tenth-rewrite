from setuptools import setup

package_name = 'rosmeasure'

setup(
    name=package_name,
    version='0.0.1',
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
            # 'xxx = rosmeasure.xxx:main',
        ],
    },
)
