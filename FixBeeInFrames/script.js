
const packagesContainer = document.getElementById("data-packages-container")

let dataPackagesSize = 51

for(let i=0; i<dataPackagesSize; i++) {
    const dataPackage = document.createElement("package-details")
    dataPackage.id = `${i}`
    packagesContainer.appendChild(dataPackage)
}
