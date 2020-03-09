#index of objects can change, except the unknown person 0
#names of objects cannot change
#face list order can change, known faces names must exist in the general category index

se_demo_ci = {}

se_demo_ci[171] = {}
se_demo_ci[171]['name'] = 'mug_cup' #Mug

se_demo_ci[179] = {}
se_demo_ci[179]['name'] = 'mug_cup' #Coffee cup

se_demo_ci[286] = {}
se_demo_ci[286]['name'] = 'bottle' #Bottle

se_demo_ci[55] = {}
se_demo_ci[55]['name'] = 'laptop' #Laptop

se_demo_ci[447] = {}
se_demo_ci[447]['name'] = 'book' #Book

se_demo_ci[135] = {}
se_demo_ci[135]['name'] = 'computer' #Computer monitor

se_demo_ci[313] = {}
se_demo_ci[313]['name'] = 'cell_phone' #Mobile phone

se_demo_ci[121] = {}
se_demo_ci[121]['name'] = 'table' #Desk

se_demo_ci[281] = {}
se_demo_ci[281]['name'] = 'table' #Table

se_demo_ci[97] = {}
se_demo_ci[97]['name'] = 'chair' #Chair

se_demo_ci[485] = {}
se_demo_ci[485]['name'] = 'window' #Window

se_demo_ci[502] = {}
se_demo_ci[502]['name'] = 'face' #Human face

#se_demo_ci[69] = {}
#se_demo_ci[69]['name'] = 'person_unknown' #Person

# se_demo_ci[292] = {}
# se_demo_ci[292]['name'] = 'person_unknown' #Human head

#se_demo_ci[308] = {}
#se_demo_ci[308]['name'] = 'person_unknown' #Man

#se_demo_ci[228] = {}
#se_demo_ci[228]['name'] = 'person_unknown' #Woman

se_demo_ci[0] = {}
se_demo_ci[0]['name'] = 'person_unknown' # to set unknown faces


svm_face_list = ['Alejandro Toledo',
 'Alvaro Uribe',
 'Amelie Mauresmo',
 'Andre Agassi',
 'Angelina Jolie',
 'Ariel Sharon',
 'Arnold Schwarzenegger',
 'Atal Bihari Vajpayee',
 'Bill Clinton',
 'Carlos Menem',
 'Colin Powell',
 'David Beckham',
 'Donald Rumsfeld',
 'George Robertson',
 'George W Bush',
 'Gerhard Schroeder',
 'Gloria Macapagal Arroyo',
 'Gray Davis',
 'Guillermo Coria',
 'Hamid Karzai',
 'Hans Blix',
 'Hugo Chavez',
 'Igor Ivanov',
 'Jack Straw',
 'Jacques Chirac',
 'Jean Chretien',
 'Jennifer Aniston',
 'Jennifer Capriati',
 'Jennifer Lopez',
 'Jeremy Greenstock',
 'Jiang Zemin',
 'John Ashcroft',
 'John Negroponte',
 'Jose Maria Aznar',
 'Juan Carlos Ferrero',
 'Junichiro Koizumi',
 'Keanu Reeves',
 'Kofi Annan',
 'Laura Bush',
 'Lindsay Davenport',
 'Lleyton Hewitt',
 'Luiz Inacio Lula da Silva',
 'Mahmoud Abbas',
 'Megawati Sukarnoputri',
 'Michael Bloomberg',
 'Naomi Watts',
 'Nestor Kirchner',
 'person_PanosGnk',
 'person_PanosPet',
 'Paul Bremer',
 'Pete Sampras',
 'Recep Tayyip Erdogan',
 'Ricardo Lagos',
 'Roh Moo-hyun',
 'Rudolph Giuliani',
 'Saddam Hussein',
 'Serena Williams',
 'Silvio Berlusconi',
 'Tiger Woods',
 'Tom Daschle',
 'Tom Ridge',
 'Tony Blair',
 'Vicente Fox',
 'Vladimir Putin',
 'Winona Ryder']