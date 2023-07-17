// зайти на сайт https://www.inaturalist.org/projects/strekozy-moskovskoy-oblasti-dragonflies-of-the-moscow-region-russia-815bf252-ee71-4519-840f-1620fd207bc5?tab=species
// ввести код в консоли
var jsonString = [];
var elements = document.getElementsByClassName("sciname species secondary-name");

for (const element of elements) {
    let jsonElem = {
        href: element.href,
        name: element.href.split("/")[4].split("-").slice(1, 3).join(" ")
    }

    jsonString.push(jsonElem)
}

JSON.stringify(jsonString)

