import * as THREE from 'three';
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

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

    const ambientLight = new THREE.AmbientLight(0x404040, 1);
    scene.add(ambientLight);
    const dirlight = new THREE.DirectionalLight(0xffffff, 0.4);
    dirlight.position.set(1, 1, 0);
    camera.add(dirlight); 
    scene.add(camera); 

    let loadedObject;

    const loader = new OBJLoader();
    loader.load(
        assetPath,
        (object) => {
            const boundingBox = new THREE.Box3().setFromObject(object);
            const center = new THREE.Vector3();
            boundingBox.getCenter(center);
            object.position.sub(center);

            const size = new THREE.Vector3();
            boundingBox.getSize(size);
            const maxDim = Math.max(size.x, size.y, size.z);
            const desiredSize = 3; 
            const scale = desiredSize / maxDim;
            object.scale.set(scale, scale, scale);
            
            scene.add(object);
            loadedObject = object;
            initGUI(container, object.children[0]);

            renderer.render(scene, camera);
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

        if (loadedObject) {
            loadedObject.rotation.x += 0.01;
            loadedObject.rotation.y += 0.01;
        }

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

function initGUI(container, model) {
    const gui = new GUI({ autoPlace: false });
    gui.add(model.position, 'x', -1, 1);
    gui.add(model.position, 'y', -1, 1);
    gui.add(model.position, 'z', -1, 1);

    container.appendChild(gui.domElement);
    gui.domElement.style.position = 'absolute';
    gui.domElement.style.top = '48px';
    gui.domElement.style.right = '0px';
}


initViewer('container1', '../assets/assignment1/subdivision_cube_3.obj');
initViewer('container2', '../assets/assignment1/subdivision_bunny_1.obj');
initViewer('container3', '../assets/assignment1/decimation_bunny_2000.obj');
initViewer('container4', '../assets/assignment1/decimation_bunny_500.obj');
initViewer('container5', '../assets/assignment1/decimation_bunny_2000_subset.obj');
