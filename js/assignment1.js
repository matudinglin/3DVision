import * as THREE from 'three';
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';


function initViewer(containerId, assetPath) {
    // Get the container element where the viewer will be appended
    const container = document.getElementById(containerId);
    container.style.position = 'relative';

    // Create a new THREE.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    // Set up the camera
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / (window.innerHeight * 0.5), 0.1, 1000);
    camera.position.z = 5;

    // Set up the WebGL renderer
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight * 0.5);
    container.appendChild(renderer.domElement);

    // Set up orbit controls for the camera
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.minDistance = 2;
    controls.maxDistance = 10;
    controls.addEventListener('change', () => renderer.render(scene, camera));

    // Add lighting to the scene
    scene.add(new THREE.AmbientLight(0x404040, 2));
    const dirlight = new THREE.DirectionalLight(0xffffff, 0.5);
    dirlight.position.set(0, 0, 1);
    scene.add(dirlight);

    // Load the 3D model
    const loader = new OBJLoader();
    loader.load(
        assetPath,
        (object) => {
            const model = object.children[0];
            model.material = new THREE.MeshPhongMaterial({ color: 0x999999 });
            model.position.set(0, 0, 0);
            scene.add(model);
            initGUI(container, model);
        },
        (xhr) => console.log(`${(xhr.loaded / xhr.total * 100).toFixed(2)}% loaded`),
        (error) => console.error('An error happened', error)
    );

    // Set up the stats panel
    const stats = new Stats();
    stats.showPanel(0);
    stats.dom.style.position = 'absolute';
    stats.dom.style.top = '0px';
    container.appendChild(stats.dom);

    // Animation loop to render the scene and update stats
    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
        stats.update();
    }

    animate();

    // Adjust camera and renderer size when the window is resized
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

initViewer('container1', '../assets/assignment1/cube_subdivided.obj');
initViewer('container2', '../assets/assignment1/cube_decimated.obj');