function changeSection(section){
	let possible = ["bio", "hiring", "publications"];
	// SECTION
	for (i=0;i<possible.length();i++){
		document.getElementById(possible[i]).style.display = "none";
	}
	document.getElementById(section).style.display = "initial";
	// BUTTON COLORS
	for (i=0;i<possible.length();i++){
		let name = possible[i].concat("-button");
		document.getElementById(name).classList.add('button-outline');
	}
	document.getElementById(section.concat("-button")).classList.remove('button-outline');
}

