# Коллекция ссылок по темам УмныйДом/IoT/EmbeddedLanguages
###### Переделка этого всего в небольшой обзор - в процессе.
Сегодня IoT это уже не просто баззаорд, а отрасль индустрии, такие компании как 
IBM, Google, Amazon, Microsoft, ARM Ldt вкладываются в развитие технологий интерента вещей.

Сами по себе принтеры, холодильники и кофеварки с доступом в интеренет есть уже довольно 
давно, и их даже [преследуют](http://www.audioholics.com/news/researchers-mpaa-riaa-printer-p2p-file-sharing) по закону
различные [скользкие](https://www.riaa.com) [личности](http://www.mpaa.org).

Если уйти в сторону от довольно упортых идей типа [лампочек](https://www.extremetech.com/electronics/163972-philips-hue-led-smart-lights-hacked-whole-homes-blacked-out-by-security-researcher), управляемых через интернет, останутся системы распределенного мониторинга и управления, системы автоматизации зданий, и сопутствующие технологии.

## Умный дом.
О системах автоматизации зданий говорят уже довольно давно (с тех времен, когда про IoT еще не слышали).
Современная система автоматизации здания - законченное решение, в том числе включающее в себя возможности интернета 
вещей (сервер с "веб-мордой", дектопные и мобильные клиенты).
Кроме того там, как правило, есть компоненты для интеграции различного оборудования умного дома 
(датчиков и исполнителей), и обязательно средства для реализации алгоритмов управления.
Ниже небольшой обзор открытых систем автоматизации зданий.

### [openHAB](http://www.openhab.org/)
Наиболее известная и распиаренная система автоматизации зданий. Написана на JAVA, переносима, 
поддерживает огромное количество устройств (датчиков и исполнителей) и коммуникационных протоколов.
Есть приложения для iOS и Andriod, web-интерфейс, простые средства разработки, более подробно
[тут](https://habrahabr.ru/post/232969/).

### [Calaos](https://calaos.fr)
Когда-то - коммерческий продукт, однако после закрытия одноименной компании исходный код был опубликован под GPL.
Система разрабатывалась, как законченное решение. Включает в себя сервер, web-интерфейс, мобильные клиенты, готовый образ GNU/Linux.
Написано главным образом на C++.

### [Domoticz](https://domoticz.com/)
Написана на C/C++, лицензия GPL, переносима, работает на большом количестве платформ и ОС.
Предоставляет web-интерфейс на HTML5. Поддерживает большое количество устройств.

### Хотите еще?
Тогда читайте:
* https://opensource.com/life/16/3/5-open-source-home-automation-tools
* http://www.hometoys.com/article/2015/10/nine-open-source-home-automation-projects/32466

## SCADA
SCADA - системы использовались в территориальнораспределенных системах управления и диспетчеризации, когда до интеренета вещей было ещё очень далеко.

### [Proview](http://www.proview.se/doc/en_us/qguide_f.html)
Одна из первых свободных SCADA. Лицензия GPL. Поддерживает Profibus DP, Modbus, и т.д. 
Позволяет создавать softPLC на графических языках IEC 61131-3 и на ЯП общего назначения.

### [ScadaBR](https://sourceforge.net/projects/scadabr/)
Бразильская SCADA. Лицензия GPL. Языки JS, PHP, C\#, Java. Предоставляет web-API для интеграции с другим ПО.
Доступ к запущенному приложению через web-интерфейс.

### [openSCADA](http://openscada.org/)
Часть проекта Eclipse. Язык JAva. Есть web-Интерфейс, архив, графики, скриптовый движок.

### [Stantor](http://stantor.free.fr/indexp_EN.htm)
Французская свободная SCADA. Лицензия GPL. web-Интерфейс, архив, графики, скриптовый движок. Позволяет управлять устройствами по сети и через USB.

### Хотите еще?
Тогда читайте:
* http://linuxscada.ru/
* http://linuxscada.info/
* http://electronicsforu.com/resources/cool-stuff-misc/8-free-open-source-software-scada


## IoT

### Обзор платформ.
* https://www.postscapes.com/internet-of-things-award/open-source/
* https://www.postscapes.com/internet-of-things-platforms/
* https://theiotlist.com/resources/

### Платформы/Проекты
* https://www.slideshare.net/IanSkerrett/using-open-source-for-iot-44049876
* https://iot.eclipse.org/java/
* http://www.eclipse.org/om2m/
* https://github.com/kaaproject/kaa
* http://www.openiot.eu/

### Платформы для создания девайсов
#### Открытые
* C/C++ IoT            https://riot-os.org/#features
* С++ IoT              https://www.mbed.com/en/
* Python IoT           http://docs.micropython.org/en/latest/pyboard/library/network.html
* LUA ???              http://www.eluaproject.net/
* LUA IoT              http://www.nodemcu.com/index_en.html
* JS ???               https://www.espruino.com/
* C/C++ Python JS IoT  https://cesanta.com/about.html
* C\#                  https://github.com/NETMF
#### Закрытые
* Java IoT   http://www.microej.com/
* Python IoT https://www.zerynth.com/



