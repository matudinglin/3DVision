import * as THREE from 'three';
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

function initViewer(containerId, assetPath) {
    const container = document.getElementById(containerId);
    container.style.position = 'relative';

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / (window.innerHeight * 0.5), 0.1, 1000);
    camera.position.z = 5;

    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight * 0.5);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.minDistance = 2;
    controls.maxDistance = 10;

    let loadedObject;

    const loader = new PLYLoader();
    loader.load(
        assetPath,
        (geometry) => {
            geometry.computeVertexNormals();
            
            if (geometry.attributes.color) {
                const material = new THREE.PointsMaterial({ vertexColors: true, size: 0.01 });
                const pointCloud = new THREE.Points(geometry, material);

                const boundingBox = new THREE.Box3().setFromObject(pointCloud);
                const center = new THREE.Vector3();
                boundingBox.getCenter(center);
                pointCloud.position.sub(center);

                const size = new THREE.Vector3();
                boundingBox.getSize(size);
                const maxDim = Math.max(size.x, size.y, size.z);
                const desiredSize = 3;
                const scale = desiredSize / maxDim;
                pointCloud.scale.set(scale, scale, scale);

                scene.add(pointCloud);
                loadedObject = pointCloud;
                initGUI(container, pointCloud);

                renderer.render(scene, camera);
            } else {
                console.warn('PLY file does not contain color information.');
            }
        },
        (xhr) => console.log(`${(xhr.loaded / xhr.total * 100).toFixed(2)}% loaded`),
        (error) => console.error('An error happened', error)
    );

    const stats = new Stats();
    stats.showPanel(0);
    stats.dom.style.position = 'absolute';
    stats.dom.style.top = '0px';
    container.appendChild(stats.dom);

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
        stats.update();
    }

    animate();

    function onWindowResize() {
        camera.aspect = window.innerWidth / (window.innerHeight * 0.5);
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight * 0.5);
    }
    window.addEventListener('resize', onWindowResize, false);
}

function initGUI(container, pointCloud) {
    const gui = new GUI({ autoPlace: false });
    gui.add(pointCloud.position, 'x', -1, 1);
    gui.add(pointCloud.position, 'y', -1, 1);
    gui.add(pointCloud.position, 'z', -1, 1);

    container.appendChild(gui.domElement);
    gui.domElement.style.position = 'absolute';
    gui.domElement.style.top = '48px';
    gui.domElement.style.right = '0px';
}


initViewer('container1', '../assets/assignment2/fountain-P11/point-clouds/cloud_11_view.ply');
initViewer('container2', '../assets/assignment2/cloud_8_view.ply');