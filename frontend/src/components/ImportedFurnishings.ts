import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import type { TwoStoryHouseModel } from "./TwoStoryHouse";
import { STORY_HEIGHT } from "./TwoStoryHouse";

type FurniturePlacement = {
  id: string;
  label: string;
  floor: "ground" | "upper";
  position: [number, number, number];
  rotation: number;
  targetSpan: number;
  targetHeight?: number;
  placeholder?: string;
};

const placements: FurniturePlacement[] = [
  {
    id: "sofa_02",
    label: "linen sofa",
    floor: "ground",
    position: [3.55, 0, -3.62],
    rotation: Math.PI,
    targetSpan: 3.0,
    placeholder: "procedural-sofa",
  },
  {
    id: "modern_coffee_table_01",
    label: "walnut coffee table",
    floor: "ground",
    position: [3.55, 0, -1.58],
    rotation: 0.08,
    targetSpan: 1.58,
    placeholder: "procedural-coffee-table",
  },
  {
    id: "modern_arm_chair_01",
    label: "mismatched armchair",
    floor: "ground",
    position: [5.45, 0, -2.1],
    rotation: -0.58,
    targetSpan: 1.18,
  },
  {
    id: "dining_table",
    label: "worn dining table",
    floor: "ground",
    position: [1.15, 0, 2.75],
    rotation: 0.035,
    targetSpan: 2.0,
    placeholder: "procedural-dining-table",
  },
  {
    id: "wooden_bookshelf_worn",
    label: "worn study bookshelf",
    floor: "ground",
    position: [6.0, 0, 1.58],
    rotation: Math.PI,
    targetSpan: 1.78,
    targetHeight: 2.12,
    placeholder: "procedural-study-bookshelf",
  },
  {
    id: "rubber_boots",
    label: "muddy entry boots",
    floor: "ground",
    position: [-5.18, 0, 3.55],
    rotation: 0.18,
    targetSpan: 0.5,
  },
  {
    id: "dirty_football",
    label: "scuffed football",
    floor: "ground",
    position: [-3.65, 0, 3.7],
    rotation: -0.28,
    targetSpan: 0.3,
  },
  {
    id: "office_notepads",
    label: "open work notes",
    floor: "ground",
    position: [4.58, 0.835, 3.47],
    rotation: -0.18,
    targetSpan: 0.48,
  },
  {
    id: "wicker_basket_01",
    label: "upstairs laundry basket",
    floor: "upper",
    position: [2.48, STORY_HEIGHT, 0.02],
    rotation: 0.22,
    targetSpan: 0.62,
  },
  {
    id: "vintage_day_bed",
    label: "vintage guest bed",
    floor: "upper",
    position: [-4.5, STORY_HEIGHT, -2.7],
    rotation: Math.PI,
    targetSpan: 2.25,
    placeholder: "procedural-second-bed",
  },
  {
    id: "painted_wooden_cabinet_02",
    label: "chipped bedroom cabinet",
    floor: "upper",
    position: [5.62, STORY_HEIGHT, -0.18],
    rotation: Math.PI,
    targetSpan: 1.72,
    targetHeight: 2.08,
    placeholder: "procedural-master-wardrobe",
  },
];

function disposeObject(root: THREE.Object3D): void {
  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    object.geometry.dispose();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      const mapped = material as THREE.MeshStandardMaterial;
      mapped.map?.dispose();
      mapped.normalMap?.dispose();
      mapped.roughnessMap?.dispose();
      mapped.metalnessMap?.dispose();
      material.dispose();
    }
  });
}

function normalizeModel(
  source: THREE.Group,
  placement: FurniturePlacement,
): THREE.Group {
  source.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(source);
  const size = bounds.getSize(new THREE.Vector3());
  const horizontalSpan = Math.max(size.x, size.z);
  const scale = placement.targetHeight && size.y > 0
    ? placement.targetHeight / size.y
    : horizontalSpan > 0
      ? placement.targetSpan / horizontalSpan
      : 1;
  source.scale.setScalar(scale);
  source.updateMatrixWorld(true);

  const scaledBounds = new THREE.Box3().setFromObject(source);
  const center = scaledBounds.getCenter(new THREE.Vector3());
  source.position.set(-center.x, -scaledBounds.min.y, -center.z);

  source.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    object.castShadow = true;
    object.receiveShadow = true;
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      const mapped = material as THREE.MeshStandardMaterial;
      if (mapped.map) mapped.map.anisotropy = 8;
      mapped.needsUpdate = true;
    }
  });

  const anchor = new THREE.Group();
  anchor.name = `imported-${placement.id}`;
  anchor.userData.assetLabel = placement.label;
  anchor.position.set(...placement.position);
  anchor.rotation.y = placement.rotation;
  anchor.add(source);
  return anchor;
}

export function attachImportedFurnishings(house: TwoStoryHouseModel): () => void {
  const loader = new GLTFLoader();
  let cancelled = false;

  for (const placement of placements) {
    const url = `/assets/models/${placement.id}/${placement.id}.gltf`;
    void loader.loadAsync(url).then(({ scene }) => {
      if (cancelled) {
        disposeObject(scene);
        return;
      }

      const imported = normalizeModel(scene, placement);
      const parent = placement.floor === "upper" ? house.upperFloor : house.groundFloor;
      parent.add(imported);
      if (placement.placeholder) {
        const fallback = house.root.getObjectByName(placement.placeholder);
        if (fallback) fallback.visible = false;
      }
    }).catch((error: unknown) => {
      console.warn(`Could not load ${placement.label}; retaining the local fallback.`, error);
    });
  }

  return () => {
    cancelled = true;
  };
}
