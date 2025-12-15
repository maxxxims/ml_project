start_service:
	docker-compose up --build -d
	echo "http://localhost:8000/docs"

stop_service:
	docker-compose down

clean_searvice:
	docker-compose down -v


ml_pipline:
	docker-compose up ml_runner --build -d
	docker-compose exec ml_runner bash -c "python -m ml_pipline.pipline --max-train-iterations 100 --max-df-size 1000"